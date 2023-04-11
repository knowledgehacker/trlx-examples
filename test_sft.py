import torch
from datasets import load_dataset

import config as cfg
from model_loader import get_tokenizer, prepare_merged_model#, load_pretrained_model


def test_batch(top_n):
    post_list = []
    dataset = load_dataset(cfg.SUMMARIZATION_DATASET, split="valid[:%d]" % top_n)
    for sample in dataset:
        post_list.append((sample["prompt"], sample["label"]))

    with torch.cuda.amp.autocast():
        for prompt, label in post_list:
            outputs = model.generate(**tokenizer(prompt, return_tensors="pt"), max_new_tokens=100)
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("prompt: %s\n" % prompt)
            print("output: %s\n" % output)
            print("label: %s\n" % label)


if __name__ == "__main__":
    tokenizer = get_tokenizer(cfg.PT_MODEL)
    # model = load_pretrained_model(cfg.PT_MODEL)
    model = prepare_merged_model(cfg.SFT_CKPT_DIR)
    model.eval()

    test_batch(1)
