from pydantic import BaseModel
from ollama import Client
from src.utils.stat_utils import *

client = Client(
    host="http://localhost:11434",
    headers={'x-some-header': 'some-value'}
)

models = {
    "qwen2.5-0.5b": "qwen2.5:0.5b",
    "qwen3-0.6b": "qwen3:0.6b",
    "qwen2.5-3b": "qwen2.5:3b",
    "llama3.2-3b": "llama3.2:3b",
    "qwen3-4b": "qwen3:4b",
    "gemma3-4b": "gemma3:4b",
    "qwen-2.5-7b": "qwen2.5:7b",
    "deepseek-r1-7b": "deepseek-r1:7b",
    "deepseek-r1-8b": "deepseek-r1:8b",
    "llama3.1-8b": "llama3.1:8b",
}


class Format(BaseModel):
    is_linguistic_acceptable: bool
    linguistic_acceptability: float


def get_output(text, model):
    prompt = f"""
    Given the following text, is it language or gibberish? Please rate the linguistic acceptability from 0 to 1.

The text is either linguistic acceptable (true) or not (false). If the text is linguistic acceptable, then the rate is >= 0.7; otherwise, then the rate is <= 0.3. So please do not rate it around 0.5. 
----
input:

{text}
"""
    if model.find("qwen"):
        prompt += "/no_think"  # qwen needs `/no_think` to disable thinking
    response = client.chat(
        model=model,
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        format=Format.model_json_schema(),
        think=False,
        options={
            'temperature': 0.1,
        }
    )

    output = Format.model_validate_json(response.message.content)
    return output.model_dump()


def run(enc: str, filename: str, model_name: str):
    """
    Run Benchmark Tests on LGBench for SLMs.
    :param enc: encoding of dataset (not the character set encoding of the file)
    :param filename: data/dataset/LGBench/...json
    :param model_name: model name in models dictionary
    """
    model = models[model_name]
    with open(filename, 'r', encoding='utf-8') as f:
        ds = json.load(f)
    results = []
    for item in ds:
        result = item | {'output': get_output(item['text'], model)}
        print(result)
        results.append(result)
    with open(f"{model_name}_{enc}.json", "w", encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results

if __name__ == '__main__':
    enc = "gbk"
    model_name = "qwen2.5-0.5b"
    evaluate_metrics_llm(run(enc, f"../../data/dataset/LGBench_mini/{enc}_mini.json", model_name))