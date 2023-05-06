import torch
#torch.set_default_dtype(torch.float16)

#import evaluate
from sft.summarize_dataset import create_summarization_dataset, TLDRDataset
from transformers import (
    #AutoTokenizer,
    Trainer,
    TrainingArguments,
    #default_data_collator,
    DataCollatorForLanguageModeling
)

import config as cfg
from model_loader import get_tokenizer, load_pretrained_model_in_8bit, prepare_peft_model_for_training#, _merge_adapter_layers

if __name__ == "__main__":
    """
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

    print("Load dataset and tokenize...")

    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.PT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    """
    tokenizer = get_tokenizer(cfg.PT_MODEL)

    # TODO: can we access summarization data in batch???
    # Set up the datasets
    train_posts = create_summarization_dataset(cfg.SUMMARIZATION_DATASET, 10000, "train")
    train_dataset = TLDRDataset(
        train_posts,
        tokenizer,
        max_length=cfg.MAX_SUM_LEN,
    )
    """
    valid_posts = create_summarization_dataset(cfg.SUMMARIZATION_DATASET, 1000, "valid")
    valid_dataset = TLDRDataset(
        valid_posts,
        tokenizer,
        max_length=cfg.MAX_SUM_LEN,
    )
    """

    print("Prepare PEFT model...")

    # load pretrained model in int8 precision and fine tune using low rank adaption
    #model = AutoModelForCausalLM.from_pretrained(cfg.PT_MODEL, use_cache=False)
    pretrained_model = load_pretrained_model_in_8bit(cfg.PT_MODEL)
    peft_model = prepare_peft_model_for_training(pretrained_model)

    print("Fine tuning...")

    output_dir = cfg.SFT_CKPT_DIR

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=output_dir,
        fp16=True,
        half_precision_backend="cuda_amp",#"apex",
        ### train
        num_train_epochs=1,# 3,
        warmup_steps=100,# lr scheduler
        gradient_accumulation_steps=cfg.SFT_GRAD_ACCU,
        # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        # gradient_checkpointing=True,
        ### evaluation
        # evaluation_strategy="steps",
        # eval_steps=500,
        # eval_accumulation_steps=1,
        #load_best_model_at_end=True,
        ### logging
        #logging_dir="./logs",# output_dir/runs/..., by default
        logging_steps=50,
        report_to=None# "none",
        # deepspeed=cfg.SFT_DS_CFG
    )
    training_args.set_dataloader(
        train_batch_size=cfg.SFT_TRAIN_MINI_BATCH_SIZE,#=per_device_train_batch_size
        #eval_batch_size=1,#=per_device_eval_batch_size
        num_workers=4,
        pin_memory=True,
    )
    training_args.set_optimizer(
        name="adamw_hf",#"adamw_apex_fused",
        learning_rate=1e-5,# initial learning rate, learning rate changes according to lr scheduler during train
        # beta1=0.9,
        # beta2=0.95,
    )
    training_args.set_save(
        strategy="steps",
        steps=50,  # 1000,
        total_limit=1,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        #eval_dataset=valid_dataset,
        #compute_metrics=compute_metrics,
        #data_collator=default_data_collator,
        #preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    peft_model.config.use_cache = False
    trainer.train()
    """
    with torch.autocast("cuda"):
        trainer.train()
    """
    model_dir = cfg.SFT_MODEL_DIR
    print("Save sft model's adapter layers to directory %s" % model_dir)
    peft_model.save_pretrained(model_dir)
