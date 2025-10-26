import json
import random
from collections import defaultdict
from typing import List, Dict, Any

random.seed(42)


def extract_samples(pos_file, neg_file, samples_per_length: int = 100, pos_neg_ratio: tuple = (1, 1),
                    len_range: tuple = (4, 16)):
    with open(pos_file, 'r', encoding='utf-8') as f:
        pos_data = json.load(f)
    with open(neg_file, 'r', encoding='utf-8') as f:
        neg_data = json.load(f)

    pos = defaultdict(list)
    neg = defaultdict(list)
    for record in pos_data:
        len_bytes = record['bytes'].count("\\x")
        if len_range[0] <= len_bytes <= len_range[1] and len_bytes % 2 == 0:
            pos[len_bytes].append(record)
    for record in neg_data:
        len_bytes = record['bytes'].count("\\x")
        if len_range[0] <= len_bytes <= len_range[1] and len_bytes % 2 == 0:
            neg[len_bytes].append(record)

    # Randomly sample a specified number of samples for each length
    sampled_list = []

    for length, records in pos.items():
        sampled_records = random.sample(records, int(samples_per_length * (pos_neg_ratio[0] / sum(pos_neg_ratio))))

        for record in sampled_records:
            sampled_list.append(
                {
                    'text': record['text'],
                    'bytes': record['bytes'],
                    'enc': record['enc'],
                    'label': True
                }
            )
    print("pos length", len(sampled_list))
    for length, records in neg.items():
        sampled_records = random.sample(records, int(samples_per_length * (pos_neg_ratio[1] / sum(pos_neg_ratio))))

        for record in sampled_records:
            sampled_list.append(
                {
                    'text': record['text'],
                    'bytes': record['bytes'],
                    'enc': record['enc'],
                    'label': False
                }
            )
    print("size", len(sampled_list))
    # group by bytes length
    grouped_samples = defaultdict(list)

    for sample in sampled_list:
        # cal bytes length ("\\xXX" is a byte)
        byte_length = sample['bytes'].count("\\x")
        grouped_samples[byte_length].append(sample)
    # Shuffle samples within each length group, then merge by sorting by length
    sorted_samples = []
    for length in sorted(grouped_samples.keys()):
        group = grouped_samples[length]
        random.shuffle(group)
        sorted_samples.extend(group)
    res = [{'id': i, **rec} for i, rec in enumerate(sorted_samples, start=1)]

    return res


def save_samples(samples: List[Dict[str, Any]], output_filename: str):
    """
    save samples to a json file
    """
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def extract_samples_by_rate(filename, ratio):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_count = len(data)
    sample_count = int(total_count * ratio)

    sampled_data = random.sample(data, sample_count)
    print("total count:", total_count, "sample count:", sample_count)

    return sampled_data


# example
if __name__ == '__main__':
    enc = "shift_jis"
    tag = "neg"
    save_samples(extract_samples(f"../../data/all_samples/{enc}_pos.json",
                                 f"../../data/all_samples/{enc}_neg.json",
                                 samples_per_length=100, # if mini, set to 50
                                 pos_neg_ratio=(1, 1)
                                 ),
                 f"../../data/dataset/LGBench/{enc}.json")
