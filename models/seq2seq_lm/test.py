from transformers import AutoConfig, AutoTokenizer


def main():

    MODEL_CONFIG = AutoConfig.from_pretrained("facebook/bart-large")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    pad_token_id = tokenizer.pad_token
    print(tokenizer.convert_tokens_to_ids(pad_token_id))


if __name__ == "__main__":
    main()
