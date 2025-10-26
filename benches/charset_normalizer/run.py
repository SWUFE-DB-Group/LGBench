import charset_normalizer
import json

def str_to_bytes(s):
    """
    将 '\\x..' 字符串转换为 bytes
    """
    return s.encode('utf-8').decode('unicode_escape').encode('latin1')


# 应该找到负样本的 from，如果 charsetn 的预测结果与 from 不同，也算错
if __name__ == '__main__':
    enc = "shift_jis"
    with open(f"../testset/samples_{enc}_100.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    for item in data:
        byte_str = str_to_bytes(item['bytes'])
        det_res = charset_normalizer.detect(byte_str)
        output = byte_str.decode(det_res['encoding']) == item['text']
        res = {
            'output': output,
            'result':det_res
        }
        # print(item | res)
        results.append(item | res)
        if det_res['confidence']<0.8:
            print(item)
            print(res)
    with open(f"charset_n_{enc}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)