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

    batch_size = 16
    mini_batch_size = 8
    gradient_accumulation_steps = batch_size // mini_batch_size

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        learning_rate=1e-5,
        per_device_train_batch_size=mini_batch_size,
        #per_device_eval_batch_size=1,
        # evaluation_strategy="steps",
        # eval_accumulation_steps=1,
        #gradient_checkpointing=True,
        fp16=True,
        #adam_beta1=0.9,
        #adam_beta2=0.95,
        output_dir=output_dir,
        num_train_epochs=1,#5,
        warmup_steps=100,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=50,#1000,
        # eval_steps=500,
        #load_best_model_at_end=True,
        logging_steps=50,
        save_total_limit=1,
        report_to=None#"none",
        # deepspeed=cfg.SFT_DS_CFG
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
