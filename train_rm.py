import os

import torch
from rm.reward_model import OPTRewardModel
#from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

import config as cfg
from rm.comparison_dataset import create_comparison_dataset, PairwiseDataset, DataCollatorReward
from model_loader import get_tokenizer

#### initialize a reward model from the SFT model and train it to be a pairwise ranker with comparison dataset ####


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result


if __name__ == "__main__":
    output_dir = cfg.RM_CKPT_DIR
    #tokenizer = AutoTokenizer.from_pretrained(cfg.PT_MODEL)
    #tokenizer.pad_token = tokenizer.eos_token
    tokenizer = get_tokenizer(cfg.PT_MODEL)

    print("Load fine-tuned model and initialize reward model using it...")

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    #model = GPTRewardModel(cfg.REWARD_MODEL)
    model = OPTRewardModel(cfg.SFT_CKPT_DIR)

    print("Train reward model...")

    """
    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)
    """

    # Create the comparisons datasets
    data_path = cfg.COMPARISON_DATASET
    train_pairs = create_comparison_dataset(data_path, 100, "train")
    #val_pairs = create_comparison_dataset(data_path, "test")

    # Make pairwise datasets for training
    max_length = 550
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    #val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        logging_steps=10,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        #evaluation_strategy="steps",
        per_device_train_batch_size=1,
        #per_device_eval_batch_size=1,
        #eval_accumulation_steps=1,
        #eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        #deepspeed=cfg.RM_DS_CFG,
        save_total_limit=1,
        report_to="none"
    )

    Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        #compute_metrics=compute_metrics,
        #eval_dataset=val_dataset,
        data_collator=DataCollatorReward()
    ).train()

