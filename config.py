import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#PT_MODEL = "EleutherAI/gpt-j-6B"
PT_MODEL = "facebook/opt-2.7b"

# sft model stuff
SFT_DS_CFG = "sft/ds_config.json"

# https://huggingface.co/datasets/CarperAI/openai_summarize_tldr
SUMMARIZATION_DATASET = "CarperAI/openai_summarize_tldr"

#SFT_MODEL = "knowledgehacker/sft_tldr"
SFT_MODEL_DIR = "model/sft"
SFT_CKPT_DIR = "ckpt/sft"

MAX_SUM_LEN = 550
MAX_NEW_TOKENS = 50

# reward model stuff
RM_DS_CFG = "rm/ds_config.json"

# https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons
COMPARISON_DATASET = "CarperAI/openai_summarize_comparisons"

REWARD_MODEL = "knowledgehacker/rm_tldr"
RM_MODEL_DIR = "/content/model/rm"
RM_CKPT_DIR = "/content/ckpt/rm"

# ppo model stuff
PPO_MODEL = "knowledgehacker/ppo_tldr"
PPO_MODEL_DIR = "model/ppo"
#PPO_CKPT_DIR = "ckpt/ppo"
