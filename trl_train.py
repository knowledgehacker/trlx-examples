from typing import List

import torch
from tqdm import tqdm
#from transformers import pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
#from trl.core import LengthSampler

import config as cfg
from model_loader import get_tokenizer, prepare_merged_model
from util.trl_utils import build_dataset, collator, load_reward_model, get_scores
from util.token_utils import encode, decode

# tokenizer
tokenizer = get_tokenizer(cfg.PT_MODEL)

# dataset
print(">> Build dataset...")


max_input_len = cfg.MAX_SUM_LEN - cfg.MAX_NEW_TOKENS
train_dataset, train_prompt_summary_dict = build_dataset(tokenizer, cfg.SUMMARIZATION_DATASET, 2, "train", max_input_len)


# model
print(">> Prepare model...")

sft_model, merged_model = prepare_merged_model(cfg.SFT_CKPT_DIR)
model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model)
is_peft_model = getattr(model, "is_peft_model", False)
print("model is_peft_model: %s" % is_peft_model)

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
ppo_trainer = PPOTrainer(config, model, None, tokenizer=tokenizer, dataset=train_dataset, data_collator=collator, optimizer=optimizer)
"""
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
"""
# pipeline, TODO: build pipeline to incorporate reward function to generate reward score
#summarize_pipeline = pipeline("summarization", model=cfg.REWARD_MODEL, device=device)

print("Load reward model starts...")
reward_model = load_reward_model(merged_model)
reward_model.to(cfg.device)
print("Load reward model finished!")


def reward_fn(samples: List[str], **kwargs):
    original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
    original_samples = [text + train_prompt_summary_dict[text.strip()] for text in original_samples]
    original_scores = get_scores(tokenizer, reward_model, original_samples)
    scores = get_scores(tokenizer, reward_model, samples)
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
    for prompt_tensor in prompt_tensors:
        with torch.cuda.amp.autocast():
            response_with_prompt = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
        # strip prompt tokens, leaving response tokens only
        response = response_with_prompt.squeeze()[-generation_kwargs["max_new_tokens"]:]
        response_tensors.append(response)
    batch["response"] = [decode(tokenizer, r.squeeze()) for r in response_tensors]

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

    # TODO: why the whole model is checkpoint here, while only adapter layers are checkpoint in train_sft.py???
    output_dir = cfg.PPO_CKPT_DIR
    print("Checkpoint ppo model to directory %s" % output_dir)
    model.save_pretrained(output_dir)

"""
# push to huggingface hub
print("Push to hub %s" % cfg.PPO_MODEL)
model.push_to_hub(cfg.PPO_MODEL, use_auth_token=True)
"""
