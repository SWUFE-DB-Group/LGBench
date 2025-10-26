import json


def extract_all_pos(filename: str, enc: str):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    i = 1
    for item in data:
        if enc!=item["target_encoding"]:
            continue
        result = {
            "no": i,
            "text": item["text"],
            "bytes": item["bytes"],
            "enc": enc,
            "label": True
        }
        i += 1
        results.append(result)
    return results


def extract_all_neg(files: list, enc: str):
    results = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        i = 1
        for item in data:
            neg = item["decoding_results"]
            if enc not in neg:
                continue
            result = {
                "no": i,
                "text": neg[enc],
                "bytes": item["bytes"],
                "enc": enc,
                "label": False
            }
            i += 1
            results.append(result)
    return results
