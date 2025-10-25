"""
CJK Encoding Dataset Generator

This script generates a dataset for testing character encoding detection across CJK languages.
It processes translation data and creates records with encoding/decoding information.
"""

import json

# Load translation dataset
with open("trans_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize language-specific lists
sc = []  # Simplified Chinese
tc = []  # Traditional Chinese
ja = []  # Japanese
ko = []  # Korean

# Dictionary mapping language codes to their respective lists
dic = {"sc": sc, "tc": tc, "ja": ja, "ko": ko}

# Filter records by coherence score and populate language lists
for record in data:
    # Skip records with low coherence score
    if record["coherence_score"] < 0.7:
        continue
    sc.append(record["text"])
    tc.append(record["translate"]["tc"])
    ja.append(record["translate"]["ja"])
    ko.append(record["translate"]["ko"])

####################dataset####################################

import json


def create_record(no: int, text: str, target: str, cand: dict, success: int) -> dict:
    """
    Create a dataset record with encoding/decoding information.

    Args:
        no: Record ID
        text: Original text to encode
        target: Target encoding to use
        cand: Dictionary of candidate encodings to test (encoding: required_success_flag)
        success: Minimum number of successful decodings required

    Returns:
        dict: Record with encoding information, or empty dict if validation fails
    """
    # Try encoding with target encoding
    try:
        encoded = text.encode(target)
    except UnicodeEncodeError:
        return {}

    # Skip if text contains ASCII characters
    if any(ord(ch) < 128 for ch in text):
        return {}

    # Build base record
    result = {
        'id': no,
        'text': text,
        'target_encoding': target,
        'bytes': ''.join(f'\\x{byte:02x}' for byte in encoded),  # Convert bytes to hex string
        'len_text': len(text),
        'len_bytes': len(encoded),
        'decoding_results': {}
    }

    # Count successful decodings with different results
    success_count = 0

    # Test candidate encodings
    for enc, tag in cand.items():
        try:
            decoded = encoded.decode(enc)
            # Record if decoded text differs from original
            if decoded != text:
                result['decoding_results'][enc] = decoded
                success_count += 1
        except UnicodeDecodeError as e:
            # If this encoding is required to succeed (tag=1), reject the record
            if tag == 1:
                return {}

    # Reject if not enough successful decodings
    if success_count < success:
        return {}

    return result


# Configuration
TARGET_ENCODING = 'big5'  # Target character encoding for the text
LANG = 'tc'  # Language to process (tc = Traditional Chinese)

# Candidate encodings to test: 1=must succeed; 0=allowed to fail
# Note: cp932 includes shift_jis, cp949 includes euc_kr
CAND_ENCODINGS = {'gbk': 0, 'big5': 0, 'euc_kr': 0,
                  'shift_jis': 0, 'euc_jp': 0}
# 'tis-620': 0, 'koi8_r': 0, 'cp720': 0

# Remove target encoding from candidates
CAND_ENCODINGS.pop(TARGET_ENCODING)

# Results list
results = []

# Generate dataset records
i = 0  # Record ID counter
success = 1  # Minimum number of successful decodings required
len_stat = {}  # Statistics: text length distribution

for text in dic[LANG]:
    record = create_record(i, text, TARGET_ENCODING, CAND_ENCODINGS, success)
    if record:
        # Update length statistics
        len_stat.setdefault(len(text), 0)
        len_stat[len(text)] += 1
        results.append(record)
        i += 1

# Print length distribution statistics
print(len_stat)

# Save results to JSON file
output_path = f"../cjk_dataset/dataset_{TARGET_ENCODING}_{LANG}_score.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)