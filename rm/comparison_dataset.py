import json
import pandas as pd

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def create_comparison_dataset(path, top_n, split="train"):
    #dataset = load_dataset(path, split="%s[:%d]" % (split, top_percent))
    dataset = load_dataset(path, split=split)
    dataset = dataset.select(range(top_n))

    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch


class ComparisonDataset(Dataset):
    def __init__(self, comparison_path, tokenizer, max_length=550):
        with open(comparison_path, "r") as f:
            dataset = [json.loads(line) for line in f]

        self.tokenizer = tokenizer
        self.post_list = []
        self.summaries_0 = []
        self.summaries_1 = []
        self.labels = []
        self.max_length = max_length

        def make_text(post, summarize):
            return f"SUBREDDIT: r/{post['subreddit']}\nTITLE: {post['title']}\nPOST: {post['post']}\nTL;DR: {summarize}"

        for sample in dataset:  # chosen summary is always the first one
            self.post_list.append(sample["info"]["post"])
            # NOTE: The chosen summary is always the first one, i.e. `sample["summaries"][0]`
            if sample["choice"] == 0:
                self.summaries_0.append(make_text(sample["info"], sample["summaries"][0]["text"]))
                self.summaries_1.append(make_text(sample["info"], sample["summaries"][1]["text"]))
            else:
                self.summaries_0.append(make_text(sample["info"], sample["summaries"][1]["text"]))
                self.summaries_1.append(make_text(sample["info"], sample["summaries"][0]["text"]))
            self.labels.append(0)

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        summ0 = self.summaries_0[idx]
        summ1 = self.summaries_1[idx]
        encodings_dict = self.tokenizer(
            [summ0, summ1],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attention_mask = torch.tensor(encodings_dict["attention_mask"])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class AllSummDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=1024):
        df = pd.read_parquet(train_path)
        if split == "valid":
            df = df.sample(n=5000)
        self.summarizes = []
        for i, row in df.iterrows():
            self.summarizes.append(f"Summarize: {row['text']}. TL;DR: {row['summary']}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.summarizes)

    def __getitem__(self, idx):
        txt = self.summarizes[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }
