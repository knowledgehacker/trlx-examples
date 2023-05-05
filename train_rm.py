import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

import config as cfg
from model_loader import get_tokenizer, prepare_merged_model
from rm.comparison_dataset import create_comparison_dataset, PairwiseDataset, DataCollatorReward
from rm.reward_model import OPTRewardModel

#### initialize a reward model from the SFT model and train it to be a pairwise ranker with comparison dataset ####

"""
def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result
"""

if __name__ == "__main__":
    #tokenizer = AutoTokenizer.from_pretrained(cfg.PT_MODEL)
    #tokenizer.pad_token = tokenizer.eos_token
    tokenizer = get_tokenizer(cfg.PT_MODEL)

    print("Load and create comparison dataset...")

    # Create the comparisons datasets
    data_path = cfg.COMPARISON_DATASET
    train_pairs = create_comparison_dataset(data_path, 10, "train")
    #val_pairs = create_comparison_dataset(data_path, "test")

    # Make pairwise datasets for training
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=cfg.MAX_SUM_LEN)
    #val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    print("Load sft model and instantiate reward model using it...")

    # Initialize the reward model from the (supervised) fine-tuned model
    #model = OPTRewardModel(cfg.SFT_CKPT_DIR)
    _, merged_model = prepare_merged_model(cfg.SFT_MODEL_DIR)
    reward_model = OPTRewardModel(merged_model)
    reward_model.to(cfg.device)

    """
    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)
    """

    print("Train reward model...")

    output_dir = cfg.RM_CKPT_DIR

    batch_size = 128
    mini_batch_size = 4
    gradient_accumulation_steps = batch_size // mini_batch_size

    training_args = TrainingArguments(
        learning_rate=1e-5,
        per_device_train_batch_size=mini_batch_size,
        #evaluation_strategy="steps",
        #per_device_eval_batch_size=1,
        #eval_accumulation_steps=1,
        #eval_steps=500,
        fp16=True,
        output_dir=output_dir,
        num_train_epochs=5,
        warmup_steps=100,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=500,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=1,
        report_to=None#"none",
        # deepspeed=cfg.RM_DS_CFG,
    )

    trainer = Trainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        #compute_metrics=compute_metrics,
        #eval_dataset=val_dataset,
        data_collator=DataCollatorReward()
    )
    trainer.train()

    model_dir = cfg.RM_MODEL_DIR
    print("Save reward model to directory %s" % model_dir)
    trainer.save_model(model_dir)
    """
    print("Push to hub %s" % cfg.REWARD_MODEL)
    model.push_to_hub(cfg.REWARD_MODEL, use_auth_token=True)
    """