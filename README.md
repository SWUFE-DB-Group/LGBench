# LGBench

> An encoding oriented semantic validation benchmark.

This repo is mainly used for semantic encoding validation for short CJK texts (SemVal-S). And StruVal-C can be found at [Janus](https://github.com/SWUFE-DB-Group/Janus).

## How to Use for Semantic Validation

1. First download a Qwen family model (recommended: [Qwen3-0.6B](https://modelscope.cn/models/Qwen/Qwen3-0.6B) or [Qwen2.5-0.5B](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B) on [ModelScope](https://modelscope.cn/my/overview)):

   ```shell
   modelscope download --model Qwen/Qwen3-0.6B --local_dir ./your_dir
   # or
   modelscope download --model Qwen/Qwen2.5-0.5B --local_dir ./your_dir
   ```
2. You need to configure the model path: set `model_path` to the **absolute path** of your model directory in `src/nxe`. 

    ```python
   # example
   model_path = r"C:\Users\ASUS\PycharmProjects\LGBench\qwen2-0.5b"
    ```
3. Run
   ```shell
   python main.py --model <model_name> --bytes <byte_sequence> --enc <encoding>
   # --model: qwen2.5 | qwen3
   # --bytes: \x...
   # --enc:   gbk | big5 | euc-kr | euc-jp | shift-jis
   ```
   
   ```shell
   # example
   python main.py --model qwen2.5 --bytes \xcf\xc2\xb5\xa5 --enc gbk
   
   # output:   
   # decode text: 下单
   # SemVal-S result: True
   ```


## How to Run Benchmark Tests on LGBench

## Performance Reports
