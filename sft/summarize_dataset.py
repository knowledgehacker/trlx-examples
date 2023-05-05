#import json

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets import load_dataset

from util.token_utils import encode

"""
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
"""


def create_summarization_dataset(path, top_n, split):
    dataset = load_dataset(path, split="%s[:%d]" % (split, top_n))

    posts = []
    for sample in tqdm(dataset):
        posts.append(sample["prompt"] + sample["label"])
    if "valid" in split:
        posts = posts[0:2000]

    return posts


class TLDRDataset(Dataset):
    def __init__(self, posts, tokenizer, max_length):
        self.input_ids = []
        self.attention_mask = []
        for post in tqdm(posts):
            encodings_dict = encode(tokenizer, post, max_length=max_length, return_tensors="pt")
            self.input_ids.append(encodings_dict["input_ids"][0])
            self.attention_mask.append(encodings_dict["attention_mask"][0])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx], # TODO: why "labels" is self.input_ids
        }
