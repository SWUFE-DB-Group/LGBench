import json
from src.utils.openai_utils import *

# your openai base url and api key
base_url='http://localhost:11434/v1'
api_key='skxxxx'
model = "model"

INPUT_FILE = r"../../data/raw/cleaned.json" # cleaned data
OUTPUT_FILE = r"llm_output.txt"  # Raw output from LLM in txt format. Due to the model's limited structured output capability,
# this file requires manual parsing using regex to convert into JSON format.


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    batch_size = 10  # Number of records processed by LLMs per batch
    for i in range(0,len(data),batch_size):
        records = str(data[i:i+batch_size])
        prompt = f"""Translate "text" into Traditional Chinese、Japanese、Korean.
        Requirement:
          1. Please note again and again: The translation result should not contain any language other than CJK;
          2. Please retain the original meaning of the translated result. You should try your best to translate. If the translation is difficult, set "translate" to "/no_translate";
          3. "coherence_score": <0-1>float. The more fluent the translation result is in the corresponding language, the higher the "coherence_score" will be. Otherwise, the lower it is. Please make a strict judgment!
          4. Output only the final JSON objects. Do not include explanations, reasoning, or any extra text.

        input:
            {records}
        """+"""
        ---
        Examples:
        input:
            [{"id": 1, "text": "小学"},{"id": 2, "text": "批发市场"},{"id": 3, "text": "薄刀峰"}]
        output:
            [
                {"id": 1, "text": "小学", "coherence_score":0.95, "translate": {"tc":"小學", "ja":"小学校", "ko":"초등학교"}},
                {"id": 2, "text": "批发市场", "coherence_score":0.91, "translate": {"tc":"批發市場", "ja":"卸売市場", "ko":"도매시장"}},
                {"id": 3, "text": "薄刀峰", "coherence_score":0.36, "translate": {"tc":"薄刀峰", "ja":"/no_translate", "ko":"/no_translate"}}
            ]"""
        output = get_llm_output(
            base_url=base_url,
            api_key=api_key,
            model=model,
            user_prompt=prompt
        )
        # print(prompt)
        # print(output)
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(output)



if __name__ == '__main__':
    main()
