import charset_normalizer
from src.utils.enc_utils import *
from src.utils.stat_utils import *


def run(enc:str, filename:str):
    """
    Run Benchmark Tests on LGBench for Charset Normalization.
    :param enc: encoding of dataset (not the character set encoding of the file)
    :param filename: data/dataset/LGBench/...json
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    for item in data:
        byte_str = str_to_bytes(item['bytes'])
        det_res = charset_normalizer.detect(byte_str)
        output = byte_str.decode(det_res['encoding']) == item['text'] # Character sets are compatible, such as gb18030-gbk-gb2312
        # so an error is considered only if the decoding result using the detected character set is different from the original text

        res = {
            'output': output,
            'result':det_res
        }
        results.append(item | res)

    with open(f"charsetn_{enc}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return results

if __name__ == '__main__':
    enc = "shift_jis"
    evaluate_metrics_charsetn(run(enc,f"../../data/dataset/LGBench/{enc}.json"))