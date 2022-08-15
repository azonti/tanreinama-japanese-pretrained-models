import argparse
import os
from tokenization.tokenization_gpt2_japanese import (
    GPT2JapaneseTokenizer,
)
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../model/japanese-gpt2-medium"
    )
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    GPT2JapaneseTokenizer.register_for_auto_class()

    tokenizer = GPT2JapaneseTokenizer(
        vocab_file="../tokenizer/ja-bpe.txt",
        emoji_file="../tokenizer/emoji.json"
    )
    tokenizer.save_pretrained(args.save_dir)
    tokenizer.save_vocabulary(args.save_dir)

    config = GPT2Config.from_json_file("config.json")
    model = GPT2LMHeadModel.from_pretrained("pytorch_model.bin", config=config)
    assert isinstance(model, GPT2LMHeadModel)
    model.save_pretrained(args.save_dir)
