import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.stat_utils import *

# Your Qwen model path
model_path = r"../../qwen2-0.5b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float32,
    device_map=None
)
model.eval()


def calculate_perplexity(text, model, tokenizer):
    """
    calculate text‘s perplexity
    """
    inputs = tokenizer(text, return_tensors="pt")

    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))

    # Perplexity = e^(loss)
    perplexity = torch.exp(loss)
    return perplexity.item()


def run(enc: str, filename: str):
    """
    Run Benchmark Tests on LGBench for input PPL.
    :param enc: encoding of dataset (not the character set encoding of the file)
    :param filename: data/dataset/LGBench/...json
    """
    with open(filename, 'r', encoding='utf-8') as f:
        ds = json.load(f)
    results = []
    for item in ds:
        messages = [
            {'role': 'system', 'content': ''},
            {"role": "user", "content": item['text']}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        result = item | {"ppl": calculate_perplexity(text, model, tokenizer)}
        print(result)
        results.append(result)
    with open(f"ppl_{enc}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results

if __name__ == '__main__':
    enc = "gbk"
    evaluate_metrics_ppl(run(enc, f"../../data/dataset/LGBench/{enc}.json"),5000)
