# Scripts to port [tanreinama](https://github.com/tanreinama)'s Japanese pretrained models to HuggingFace Transformers

## How to port tanreinama's Japanese GPT-2 (medium) model

1. Download `ja-bpe.txt` and `emoji.json` from [tanreinama/gpt2-japanese](https://github.com/tanreinama/gpt2-japanese) to `tokenizer` directory.
2. Download the checkpoint from [tanreinama/gpt2-japanese](https://github.com/tanreinama/gpt2-japanese) to `checkpoint` directory.
3. Run the below script.

```bash
pip install -r requirements.txt
pushd src
transformers-cli convert --model_type gpt2 --tf_checkpoint ../checkpoint/model-10410000 --pytorch_dump_output . --config ../config/japanese-gpt2-medium-config.json
python -m convert_gpt2 --save_dir ../model/japanese-gpt2-medium
rm pytorch_model.bin config.json
python -m check_gpt2 --save_dir ../model/japanese-gpt2-medium
popd
```
