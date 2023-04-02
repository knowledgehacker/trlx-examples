
#PT_MODEL = "EleutherAI/gpt-j-6B"
PT_MODEL = "facebook/opt-2.7b"#"facebook/opt-6.7b"#"decapoda-research/llama-7b-hf"

# sft model stuff
SFT_DS_CFG = "sft/ds_config.json"

#SFT_MODEL = "knowledgehacker/sft_tldr"
SUMMARIZATION_DATASET = "CarperAI/openai_summarize_tldr"
SFT_CKPT_DIR = "sft/checkpoint"

# reward model stuff
RM_DS_CFG = "rm/ds_config.json"

#REWARD_MODEL = "CarperAI/rm_tldr"
# https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons
COMPARISON_DATASET = "CarperAI/openai_summarize_comparisons"
RM_CKPT_DIR = "/content/rm/checkpoint"

# load reward model from HuggingFace hub
RM_CKPT_PATH = "%s/pytorch_model.bin" % RM_CKPT_DIR
