import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import peft
from peft.utils.other import prepare_model_for_int8_training, _get_submodules
from peft import LoraConfig, get_peft_model
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


def prepare_peft_model_for_training(model_path):
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


def load_peft_model(adapter_path):
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
    model.eval()    # TODO: is it mandatory?

    return model


# TODO: merge the adapter layers into the base model’s weights, what does it mean really?
def merge_adapter_layers(model):
    lora_model = model.base_model
    key_list = [key for key, _ in lora_model.named_modules() if "lora" not in key]
    for key in key_list:
        parent, target, target_name = _get_submodules(lora_model, key)
        if isinstance(target, peft.tuners.lora.Linear):
            bias = target.bias is not None
            new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
            lora_model._replace_module(parent, target_name, new_module, target)

    return lora_model.model


def prepare_merged_model(adapter_path):
    model = load_peft_model(adapter_path)

    # merge the adapter layers into the base model’s weights
    model = merge_adapter_layers(model)

    return model
