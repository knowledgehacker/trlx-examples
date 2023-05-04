import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#PT_MODEL = "EleutherAI/gpt-j-6B"
PT_MODEL = "facebook/opt-2.7b"

# sft model stuff
SFT_DS_CFG = "sft/ds_config.json"

#SFT_MODEL = "knowledgehacker/sft_tldr"
SUMMARIZATION_DATASET = "CarperAI/openai_summarize_tldr"
SFT_CKPT_DIR = "ckpt/sft"

MAX_SUM_LEN = 550
MAX_NEW_TOKENS = 50

# reward model stuff
RM_DS_CFG = "rm/ds_config.json"

REWARD_MODEL = "knowledgehacker/rm_tldr"
# https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons
COMPARISON_DATASET = "CarperAI/openai_summarize_comparisons"
RM_CKPT_DIR = "/content/ckpt/rm"

# load reward model from HuggingFace hub or local directory
RM_CKPT_PATH = "%s/pytorch_model.bin" % RM_CKPT_DIR

PPO_MODEL = "knowledgehacker/ppo_tldr"
PPO_CKPT_DIR = "ckpt/ppo"

