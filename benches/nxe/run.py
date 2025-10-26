import src.nxe as nxe
from src.utils.stat_utils import *
import json


def run(enc: str, filename: str):
    """
    Run Benchmark Tests on LGBench for NXE.
    :param enc: encoding of dataset (not the character set encoding of the file)
    :param filename: data/dataset/LGBench/...json
    """
    with open(filename, 'r', encoding='utf-8') as f:
        ds = json.load(f)
    results = []
    for item in ds:
        result = item | {'ppl': nxe.get_nxe_ppl(item['text'])}
        print(result)
        results.append(result)
    with open(f"nxe_{enc}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    enc = "gbk"
    threshold = calculate_threshold(f"../../data/threshold_samples_25/{enc}_pos_samples.json")
    evaluate_metrics_ppl(run(enc, f"../../data/dataset/LGBench/gbk.json"), threshold)
