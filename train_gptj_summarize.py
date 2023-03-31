import random

#import evaluate
import numpy as np
import torch
from sft.summarize_dataset import TLDRDataset, get_dataset_from_jsonl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    #default_data_collator,
    DataCollatorForLanguageModeling
)

from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
import config as cfg


if __name__ == "__main__":
    output_dir = cfg.SFT_CKPT_DIR
    train_batch_size = 128
    gradient_accumulation_steps = 32
    learning_rate = 1e-5
    #eval_batch_size = 1
    #eval_steps = 500
    max_input_length = 200#550
    save_steps = 100#1000
    num_train_epochs = 3#5
    #random.seed(42)

    print("Load dataset and tokenize...")

    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.PT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.PT_MODEL,
                                              padding_side='right',
                                              use_fast=False)

    # Set up the datasets
    train_dataset = TLDRDataset(
        cfg.SUMMARIZATION_DATASET,
        tokenizer,
        "train",
        max_length=max_input_length,
    )
    """
    dev_dataset = TLDRDataset(
        data_path,
        tokenizer,
        "valid",
        max_length=max_input_length,
    )

    # Set up the metric
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        result = rouge.compute(predictions=pred_str, references=label_str)
        return result

    # Create a preprocessing function to extract out the proper logits from the model output
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)
    """

    print("Load pretrained model...")

    # load pretrained model in int8 precision and fine tune using low rank adaption
    #model = AutoModelForCausalLM.from_pretrained(cfg.PT_MODEL, use_cache=False)

    LORA_R = 4
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    loraCfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.PT_MODEL,
        load_in_8bit=True,
        device_map="auto",
    )
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, loraCfg)

    print("Fine tuning...")

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        #per_device_eval_batch_size=eval_batch_size,
        # evaluation_strategy="steps",
        # eval_accumulation_steps=1,
        #gradient_checkpointing=True,
        #half_precision_backend=True,
        fp16=True,
        #adam_beta1=0.9,
        #adam_beta2=0.95,
        num_train_epochs=num_train_epochs,
        warmup_steps=100,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        # eval_steps=eval_steps,
        #load_best_model_at_end=True,
        #logging_steps=50,
        #deepspeed=cfg.PT_DS_CFG,
        logging_steps=1,
        output_dir=output_dir,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,#train_dataset,
        #eval_dataset=dev_dataset,
        #compute_metrics=compute_metrics,
        #data_collator=default_data_collator,
        #preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    trainer.save_model(output_dir)
