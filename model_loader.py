import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import peft
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig


def get_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right', use_fast=False)

    return tokenizer


def load_pretrained_model(model_path, return_dict=True):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        return_dict=return_dict,
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


def prepare_merged_model(adapter_path):
    peft_config = PeftConfig.from_pretrained(adapter_path)
    """
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    """
    model = load_pretrained_model(peft_config.base_model_name_or_path)

    # merge with adapter layers
    model = merge_model_with_adapters(model, adapter_path)

    return model


def merge_model_with_adapters(model, adapter_path):
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    lora_model = model.base_model
    key_list = [key for key, _ in lora_model.model.named_modules() if "lora" not in key]
    for key in key_list:
        parent, target, target_name = lora_model._get_submodules(key)
        if isinstance(target, peft.tuners.lora.Linear):
            bias = target.bias is not None
            new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
            lora_model._replace_module(parent, target_name, new_module, target)

    return lora_model.model

