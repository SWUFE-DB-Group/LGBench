import src.nxe as nxe
from src.utils.enc_utils import *
import argparse

# threshold cal by calculate_threshold() in utils/stst_utils.py
threshold_qwen25_0b5 = {
    "GBK":75.633,
    "BIG5":95.508,
    "SHIFTJIS":301.259,
    "EUCKR":163.476,
    "EUCJP":164.706
}

threshold_qwen3_0b6 = {
    "GBK":8.234,
    "BIG5":18.064,
    "SHIFTJIS":41.72,
    "EUCKR":31.293,
    "EUCJP":33.852
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., qwen3 or qwen2.5)')
    parser.add_argument('--bytes', type=str, required=True,
                        help='Byte sequence in escaped format (e.g., "\\xcf\\xc2\\xb5\\xa5")')
    parser.add_argument('--enc', type=str, required=True,
                        help='Encoding type to validate the bytes against (e.g., gbk, big5, euc-kr, euc-jp, shift-jis)')

    args = parser.parse_args()
    # print(args.model)
    # print(args.bytes)
    # print(args.enc)
    if args.model not in ["qwen3", "qwen2.5"]:
        print("Model must be either qwen3 or qwen2.5!")
        exit(1)
    if not is_bytes_str(args.bytes):
        print("Bytes sequence must be in escaped format!")
        exit(1)
    if remove_non_alphanumeric(args.enc) not in ["GBK","BIG5",'SHIFTJIS','EUCKR','EUCJP']:
        print("Enc must be gbk, big5, euc-kr, euc-jp or shift-jis!")
        exit(1)

    b = str_to_bytes(args.bytes)
    enc = args.enc
    try:
        text = b.decode(enc)
    except:
        print("Encoding failed.")
        exit(1)
    ppl = nxe.get_nxe_ppl(text)
    if args.model.lower() == "qwen3":
        key = remove_non_alphanumeric(enc)
        ts = threshold_qwen3_0b6[key]
        print(f"decode text: {text}")
        print(f"SemVal-S result: {ppl <= ts}")
    elif args.model.lower() == "qwen2.5":
        key = remove_non_alphanumeric(enc)
        ts = threshold_qwen25_0b5[key]
        print(f"decode text: {text}")
        print(f"SemVal-S result: {ppl <= ts}")

if __name__ == '__main__':
    main()

