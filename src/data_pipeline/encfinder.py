"""
CJK Encoding Dataset Generator

This script generates a dataset for testing character encoding detection across CJK languages.
"""

import json

def create_record(no: int, text: str, target: str, cand: list, success: int) -> dict:
    """
    Create a dataset record with encoding/decoding information.

    Args:
        no: Record ID
        text: Original text to encode
        target: Target encoding to use
        cand: candidate encodings to test
        success: Minimum number of successful decodings required

    Returns:
        dict: Record with encoding information, or empty dict if validation fails
    """
    # Try encoding with target encoding
    try:
        encoded = text.encode(target)
    except UnicodeEncodeError:
        return {}

    # skip if text contains ASCII characters
    if any(ord(ch) < 128 for ch in text):
        return {}

    result = {
        'id': no,
        'text': text,
        'target_encoding': target,
        'bytes': ''.join(f'\\x{byte:02x}' for byte in encoded),  # Convert bytes to hex string
        'len_text': len(text),
        'len_bytes': len(encoded),
        'decoding_results': {}
    }

    # count successful decodings with different results
    success_count = 0

    # test candidate encodings
    for enc in cand:
        try:
            decoded = encoded.decode(enc)
            # record if decoded text differs from original, since it is a negative sample
            if decoded != text:
                result['decoding_results'][enc] = decoded
                success_count += 1
        except UnicodeDecodeError:
            pass

    # Reject if not enough successful decodings
    if success_count < success:
        return {}

    return result

def lang_filter(score_threshold=0.7):
    with open("../../data/raw/translations_CJK.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # initialize language-specific lists
    sc = []  # Simplified Chinese
    tc = []  # Traditional Chinese
    ja = []  # Japanese
    ko = []  # Korean

    dic = {"sc": sc, "tc": tc, "ja": ja, "ko": ko}

    # filter records by coherence score and language
    for record in data:
        # skip records with low coherence score
        if record["coherence_score"] < score_threshold:
            continue
        sc.append(record["text"])
        tc.append(record["translate"]["tc"])
        ja.append(record["translate"]["ja"])
        ko.append(record["translate"]["ko"])
    return dic


if __name__ == '__main__':
    # gbk: sc, big5: tc, euc_kr: ko, euc_jp: ja, shift_jis: ja
    TARGET_ENCODING = 'big5'
    LANG = 'tc'

    CAND_ENCODINGS = ['gbk', 'big5', 'euc_kr', 'shift_jis', 'euc_jp']
    CAND_ENCODINGS.remove(TARGET_ENCODING)


    results = []
    i = 0  # id counter
    success = 1  # minimum number of successful decodings required
    lang_dic = lang_filter()
    for text in lang_dic[LANG]:
        record = create_record(i, text, TARGET_ENCODING, CAND_ENCODINGS, success)
        if record:
            results.append(record)
            i += 1

    output_path = f"enc_{TARGET_ENCODING}_{LANG}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
