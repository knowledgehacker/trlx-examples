import torch
from datasets import load_dataset

import config as cfg
from util.model_utils import get_tokenizer, prepare_merged_model
from util.token_utils import encode, decode

max_input_len = cfg.MAX_SUM_LEN - cfg.MAX_NEW_TOKENS


def test_batch(tokenizer, model, top_n):
    post_list = []
    dataset = load_dataset(cfg.SUMMARIZATION_DATASET, split="test[:%d]" % top_n)
    for sample in dataset:
        post_list.append((sample["prompt"], sample["label"]))

    for prompt, label in post_list:
        with torch.cuda.amp.autocast():
            encodings_dict = encode(tokenizer, prompt, max_input_len, return_tensors="pt")
            response_with_prompt = model.generate(**encodings_dict, max_new_tokens=cfg.MAX_NEW_TOKENS)
            response = response_with_prompt.squeeze()[-cfg.MAX_NEW_TOKENS:]
            response = decode(tokenizer, response.squeeze())
            print("prompt: %s\n" % prompt)
            print("response: %s\n" % response)
            print("label: %s\n" % label)


if __name__ == "__main__":
    pretrained_tokenizer = get_tokenizer(cfg.PT_MODEL)
    sft_model, merged_model = prepare_merged_model(cfg.SFT_MODEL_DIR)

    """
    print("Test using merged_model")
    # merged_model.cuda()
    merged_model.eval()

    test_batch(pretrained_tokenizer, merged_model, 1)
    """
    print("Test using sft_model")
    #sft_model.cuda()
    sft_model.eval()

    test_batch(pretrained_tokenizer, sft_model, 1)
