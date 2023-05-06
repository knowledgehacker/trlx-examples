import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_SUM_LEN = 550
MAX_NEW_TOKENS = 50

#PT_MODEL = "EleutherAI/gpt-j-6B"
PT_MODEL = "facebook/opt-2.7b"#"facebook/opt-6.7b"

# sft model stuff
SFT_DS_CFG = "sft/ds_config.json"

# https://huggingface.co/datasets/CarperAI/openai_summarize_tldr
SUMMARIZATION_DATASET = "CarperAI/openai_summarize_tldr"

#SFT_MODEL = "knowledgehacker/sft_tldr"
SFT_MODEL_DIR = "model/sft"
SFT_CKPT_DIR = "ckpt/sft"

SFT_TRAIN_BATCH_SIZE = 32#128
SFT_TRAIN_MINI_BATCH_SIZE = 8#32
SFT_GRAD_ACCU = SFT_TRAIN_BATCH_SIZE // SFT_TRAIN_MINI_BATCH_SIZE

# reward model stuff
RM_DS_CFG = "rm/ds_config.json"

# https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons
COMPARISON_DATASET = "CarperAI/openai_summarize_comparisons"

REWARD_MODEL = "knowledgehacker/rm_tldr"
RM_MODEL_DIR = "/content/model/rm"
RM_CKPT_DIR = "/content/ckpt/rm"

RM_TRAIN_BATCH_SIZE = 32
RM_TRAIN_MINI_BATCH_SIZE = 8
RM_GRAD_ACCU = RM_TRAIN_BATCH_SIZE // RM_TRAIN_MINI_BATCH_SIZE

# ppo model stuff
PPO_MODEL = "knowledgehacker/ppo_tldr"
PPO_MODEL_DIR = "model/ppo"
#PPO_CKPT_DIR = "ckpt/ppo"
