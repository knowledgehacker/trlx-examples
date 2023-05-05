import json

import torch
#torch.set_default_dtype(torch.float16)
from datasets import load_dataset
from torch.utils.data import Dataset

from util.token_utils import encode


def get_dataset_from_jsonl(jsonl_file, return_summary=True):
    # if return_summary is True, return a list of posts with summary concatenated
    # if return_summary is False, return a list of posts and a list of summaries
    with open(jsonl_file, "r") as f:
        dataset = [json.loads(line) for line in f]
    post_list = []
    summary_list = []
    for d in dataset:
        if return_summary:
            post = f"SUBREDDIT: r/{d['subreddit']}\nTITLE: {d['title']}\nPOST: {d['post']}\nTL;DR: {d['summary']}"
        else:
            post = f"SUBREDDIT: r/{d['subreddit']}\nTITLE: {d['title']}\nPOST: {d['post']}\nTL;DR: "
            summary_list.append(d["summary"])
        post_list.append(post)
    if not return_summary:
        return post_list, summary_list
    return post_list


class TLDRDataset(Dataset):
    def __init__(self, train_path, top_n, tokenizer, split, max_length):
        self.post_list = []
        dataset = load_dataset(train_path, split="%s[:%d]" % (split, top_n))
        for sample in dataset:
            self.post_list.append(sample["prompt"] + sample["label"])
        if "valid" in split:
            self.post_list = self.post_list[0:2000]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        post = self.post_list[idx]
        encodings_dict = encode(self.tokenizer, post, max_length=self.max_length, return_tensors="pt")
        input_ids = encodings_dict["input_ids"][0]
        attn_masks = encodings_dict["attention_mask"][0]

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }
