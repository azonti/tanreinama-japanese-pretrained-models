import argparse
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    GPT2LMHeadModel,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../model/japanese-gpt2-medium",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.save_dir,
        trust_remote_code=True
    )
    assert isinstance(tokenizer, PreTrainedTokenizer)
    model = GPT2LMHeadModel.from_pretrained(
        args.save_dir,
    )
    assert isinstance(model, GPT2LMHeadModel)

    input_str = "天にまします我らの父よ。願わくは"
    model_input = tokenizer(input_str, return_tensors="pt")
    model_output = model.generate(
        **model_input,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1
    )
    output_str = tokenizer.decode(model_output[0].tolist())
    print(output_str)
