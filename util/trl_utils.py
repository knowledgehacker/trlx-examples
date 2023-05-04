from typing import List

import torch
from datasets import load_dataset

import config as cfg
from rm.reward_model import OPTRewardModel
from util.token_utils import encode, decode


# Note that in class TLDRDataset defined in summarize_dataset.py concatenates prompt and label when initializing
def format_prompt(tokenizer, sample, max_length):
    encodings_dict = encode(tokenizer, sample["query"], max_length)
    sample["input_ids"] = encodings_dict["input_ids"]
    sample["attention_mask"] = encodings_dict["attention_mask"]
    sample["query"] = decode(tokenizer, encodings_dict["input_ids"])

    return sample


def build_dataset(tokenizer, name, top_n, split, max_input_len):
    ds = load_dataset(name, split="%s[:%d]" % (split, top_n))
    # rename prompt to query, otherwise prompt column will be removed, and it can't be used later
    ds = ds.rename_columns({"prompt": "query"})
    ds = ds.map(lambda sample: format_prompt(tokenizer, sample, max_input_len), batched=False)
    ds.set_format(type="torch")

    # map prompt to summarization
    prompt_summary_dict = {}

    content = [(sample["query"], sample["label"]) for sample in ds]
    prompts, summaries = zip(*content)
    for i in range(len(prompts)):
        prompt_summary_dict[prompts[i]] = summaries[i]

    return ds, prompt_summary_dict


# collator
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def load_reward_model(pretrained_model):
    reward_model = OPTRewardModel(pretrained_model)
    reward_model.load_state_dict(torch.load(cfg.RM_CKPT_PATH))
    # reward_model.half()
    reward_model.eval()

    return reward_model


def get_scores(tokenizer, reward_model, samples: List[str]):
    score_list = []
    batch_size = 2
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i: i + batch_size]
        sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
        """
        encodings_dict = tokenizer(
            sub_samples,
            truncation=True,
            max_length=cfg.MAX_SUM_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        """
        encodings_dict = encode(tokenizer, sub_samples, cfg.MAX_SUM_LEN, return_tensors="pt")
        input_ids = encodings_dict["input_ids"]  # .to(device)
        attn_masks = encodings_dict["attention_mask"]  # .to(device)
        input_ids = input_ids.repeat(2, 1)
        attn_masks = attn_masks.repeat(2, 1)
        with torch.no_grad():
            sub_scores = reward_model(input_ids=input_ids, attention_mask=attn_masks)
        score_list.append(sub_scores["chosen_end_scores"])
    scores = torch.cat(score_list, dim=0)

    return scores
