import torch
import torch.utils.checkpoint as checkpoint
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from accelerate import Accelerator
from accelerate import infer_auto_device_map, init_empty_weights

import peft
from peft.utils.other import prepare_model_for_int8_training, _get_submodules
from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig


def get_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              #padding_side='right', # 'right' by default
                                              use_fast=False)

    return tokenizer


# https://huggingface.co/blog/accelerate-large-models
def _customize_device_map(model_path):
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"], dtype=torch.float32)
    device_map["model.decoder.embed_tokens"] = "cpu"
    device_map["model.decoder.embed_positions"] = "cpu"
    device_map["model.decoder.final_layer_norm"] = "cpu"
    for i in range(0, 10):
        device_map["model.decoder.layers.%d" % i] = "disk"
    for i in range(10, 32):
        device_map["model.decoder.layers.%d" % i] = 0
    device_map["lm_head"] = 0

    return device_map


def _load_pretrained_model_in_8bit(model_path, return_dict=True):
    print("Load pretrained model %s starts..." % model_path)
    """
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

    device_map = _customize_device_map(model_path)
    print("device_map: %s" % device_map)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        return_dict=return_dict,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map=device_map,#"auto",#{"": Accelerator().local_process_index},
        quantization_config=quantization_config,
        offload_folder="offload",
        offload_state_dict=True
    )
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        return_dict=return_dict,
        load_in_8bit=True,
        device_map="auto"
    )

    print("Model %s memory footprint: %d" % (model_path, model.get_memory_footprint()))

    print("Load pretrained model %s finished!" % model_path)

    return model


def prepare_peft_model_for_training(model_path):
    LORA_R = 4
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05

    TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "down_proj",
        "gate_proj",
        "up_proj",
    ]

    loraCfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = _load_pretrained_model_in_8bit(model_path)
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, loraCfg)

    return model


def _load_peft_model(adapter_path):
    peft_config = PeftConfig.from_pretrained(adapter_path)
    """
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    """

    model = _load_pretrained_model_in_8bit(peft_config.base_model_name_or_path)

    model = PeftModel.from_pretrained(model, adapter_path)
    #model.to("cuda")

    return model


# TODO: merge the adapter layers into the base model’s weights, what does it mean really?
def _merge_adapter_layers(model):
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
    model = _load_peft_model(adapter_path)

    # merge the adapter layers into the base model’s weights
    merged_model = _merge_adapter_layers(model)

    return model, merged_model
