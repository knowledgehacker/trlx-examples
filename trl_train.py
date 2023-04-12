from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
#from transformers import pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
#from trl.core import LengthSampler

import config as cfg
from model_loader import get_tokenizer, load_peft_model, merge_adapter_layers
from rm.reward_model import OPTRewardModel

# tokenizer
tokenizer = get_tokenizer(cfg.PT_MODEL)

# dataset
print(">> Build dataset...")


# Note that in class TLDRDataset defined in summarize_dataset.py concatenates prompt and label when initializing
def format_prompt(sample, max_length):
    encodings_dict = tokenizer(sample["query"],
                               truncation=True,
                               padding="max_length",
                               max_length=max_length,
                               add_special_tokens=False)
    sample["input_ids"] = encodings_dict["input_ids"]
    sample["query"] = tokenizer.decode(encodings_dict["input_ids"], skip_special_tokens=True, ).strip()

    return sample


def build_dataset(name, top_n, split, max_input_len):
    ds = load_dataset(name, split="%s[:%d]" % (split, top_n))
    # rename prompt to query, otherwise prompt column will be removed, and it can't be used later
    ds = ds.rename_columns({"prompt": "query"})
    ds = ds.map(lambda sample: format_prompt(sample, max_input_len), batched=False)
    ds.set_format(type="torch")

    # map prompt to summarization
    prompt_summary_dict = {}

    content = [(sample["query"], sample["label"]) for sample in ds]
    prompts, summaries = zip(*content)
    for i in range(len(prompts)):
        prompt_summary_dict[prompts[i]] = summaries[i]

    return ds, prompt_summary_dict


max_input_len = cfg.MAX_SUM_LEN - cfg.MAX_NEW_TOKENS
train_dataset, train_prompt_summary_dict = build_dataset(cfg.SUMMARIZATION_DATASET, 2, "train", max_input_len)


# collator
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# model
print(">> Prepare model...")

#sft_model = prepare_merged_model(cfg.SFT_CKPT_DIR)
sft_model = load_peft_model(cfg.SFT_CKPT_DIR)
pretrained_model = merge_adapter_layers(sft_model)
model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)
# construct reference model
sft_model.disable_adapter()
ref_pretrained_model = sft_model.base_model.model
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_pretrained_model)

model.gradient_checkpointing_disable = model.pretrained_model.gradient_checkpointing_disable
model.gradient_checkpointing_enable = model.pretrained_model.gradient_checkpointing_enable


config = PPOConfig(
    model_name=None,
    learning_rate=1e-5,
    log_with=None,
    mini_batch_size=2,#4,
    batch_size=2,#128,
    gradient_accumulation_steps=1,
)

# optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)


# trainer
print(">> Train...")

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer=tokenizer, dataset=train_dataset, data_collator=collator, optimizer=optimizer)
"""
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
"""
# pipeline, TODO: build pipeline to incorporate reward function to generate reward score
#summarize_pipeline = pipeline("summarization", model=cfg.REWARD_MODEL, device=device)


reward_model = OPTRewardModel(pretrained_model)
reward_model.load_state_dict(torch.load(cfg.RM_CKPT_PATH))
#reward_model.half()
reward_model.eval()


def get_scores(samples: List[str]):
    score_list = []
    batch_size = 2
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i: i + batch_size]
        sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
        encodings_dict = tokenizer(
            sub_samples,
            truncation=True,
            max_length=cfg.MAX_SUM_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings_dict["input_ids"]  # .to(device)
        attn_masks = encodings_dict["attention_mask"]  # .to(device)
        input_ids = input_ids.repeat(2, 1)
        attn_masks = attn_masks.repeat(2, 1)
        with torch.no_grad():
            sub_scores = reward_model(input_ids=input_ids, attention_mask=attn_masks)
        score_list.append(sub_scores["chosen_end_scores"])
    scores = torch.cat(score_list, dim=0)

    return scores


def reward_fn(samples: List[str], **kwargs):
    original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
    original_samples = [text + train_prompt_summary_dict[text.strip()] for text in original_samples]
    original_scores = get_scores(original_samples)
    scores = get_scores(samples)
    norms_scores = [score - original_score for score, original_score in zip(scores, original_scores)]
    return norms_scores


# We then define the arguments to pass to the `generate` function of the PPOTrainer,
# which is a wrapper around the `generate` function of the trained model.
# pad_token = '<pad>', id = 1; unk_token = bos_token = eos_token = '/s', id = 2; sep_token not supported;
# for model 'facebook/opt-6.7b'
generation_kwargs = {
    #"do_sample": True,  # sample a response from top k probabilities
    "max_new_tokens": cfg.MAX_NEW_TOKENS,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id
}

# Note that the last batch with sample < batch_size will be discarded
data_loader = ppo_trainer.dataloader
for epoch, batch in tqdm(enumerate(data_loader)):
    prompt_tensors = batch["input_ids"]

    # Get response from Causal LM
    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True

    response_tensors = []
    for prompt in prompt_tensors:
        with torch.cuda.amp.autocast():
            response_with_prompt = ppo_trainer.generate(prompt, **generation_kwargs)
        # strip prompt tokens, leaving response tokens only
        response = response_with_prompt.squeeze()[-generation_kwargs["max_new_tokens"]:]
        response_tensors.append(response)
    batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True, ) for r in response_tensors]

    # Compute score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    """
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": config.mini_batch_size}
    pipe_outputs = summarize_pipeline(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    """
    rewards = reward_fn(texts)

    # Run PPO step
    model.gradient_checkpointing_enable()
    model.pretrained_model.config.use_cache = False

    stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

"""
# push to huggingface hub
print("Push to hub %s" % cfg.PPO_MODEL)
model.push_to_hub(cfg.PPO_MODEL, use_auth_token=True)
"""
