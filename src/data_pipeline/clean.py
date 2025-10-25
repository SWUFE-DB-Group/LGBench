import json
from src.utils.openai_utils import *

# your openai base url and api key
base_url='http://localhost:11434/v1'
api_key='skxxxx'
model = "model"

INPUT_FILE = r"../../data/raw/THUOCL_all.json"  # For example ../../data/raw/THUOCL_all.json"
OUTPUT_FILE = r"llm_output.txt"  # Raw output from LLM in txt format. Due to the model's limited structured output capability,
# this file requires manual parsing using regex to convert into JSON format.

sys_prompt = "You are a Chinese text cleaning assistant."
prompt = """You are a Chinese text validation and cleaning assistant.
You will receive some JSON objects, each containing:

- id: a numeric identifier
- text: a short Chinese string

Your task:

1. Evaluate whether each "text" is a valid, meaningful, complete, and commonly used Chinese phrase.
2. Add two new fields, "is_valid" and "confidence", to each JSON object:
    - true: valid, fluent, complete, common, meaningful phrase.
    - false: invalid, incomplete, fragmentary phrase, or uncommon name of a person or place, gibberish, rare, stopword-based phrase.
    - confidence: <0–100>, the confidence level you believe your judgment is correct. Please make strict judgments!
3. Output only the final JSON objects in the same structure as the input. Do not include explanations, reasoning, or any extra text.

---
output format:
{
    "id": <int>,
    "text": <string>,
    "is_valid": <bool>,
    "confidence": <int>
}
---
input: 

"""

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    batch_size = 20  # Number of records processed by LLMs per batch

    for i in range(0, len(data), batch_size):
        prompt_data = prompt + str(data[i:i + batch_size])
        output = get_llm_output(
            base_url=base_url,
            api_key=api_key,
            model=model,
            user_prompt=prompt_data,
            sys_prompt=sys_prompt
        )
        print(prompt_data)
        print(output)
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(output)

if __name__ == '__main__':
    main()