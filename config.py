
#PT_MODEL = "EleutherAI/gpt-j-6B"
PT_MODEL = "facebook/opt-2.7b"#"facebook/opt-6.7b"#"decapoda-research/llama-7b-hf"
PT_DS_CFG = "sft/ds_config_gptj.json"

SFT_MODEL = "mlin/sft_tldr"

SUMMARIZATION_DATASET = "CarperAI/openai_summarize_tldr"
SFT_CKPT_DIR = "sft_checkpoint"

REWARD_MODEL = "CarperAI/rm_tldr"
RM_DS_CFG = "rm/ds_config_gpt_j.json"

COMPARISON_DATASET = "CarperAI/openai_summarize_comparisons"
RM_CKPT_DIR = "rm_checkpoint"

# load reward model from HuggingFace hub
RM_CKPT_PATH = "rm/%s/pytorch_model.bin" % RM_CKPT_DIR
