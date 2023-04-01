import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig


def get_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              padding_side='right',
                                              use_fast=False)
    return tokenizer


def load_pretrained_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map="auto"
    )

    return model


def prepare_peft_model(model_path):
    LORA_R = 4
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    loraCfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = load_pretrained_model(model_path)
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, loraCfg)

    return model


def prepare_model_with_adapters(adapter_path):
    peft_config = PeftConfig.from_pretrained(adapter_path)
    """
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    """
    model = load_pretrained_model(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, adapter_path)

    return model
