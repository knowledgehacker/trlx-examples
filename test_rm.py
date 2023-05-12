import torch
from torch.utils.data import DataLoader
from rm.reward_model import OPTRewardModel
from tqdm import tqdm
#from transformers import AutoTokenizer

import config as cfg
from rm.comparison_dataset import create_comparison_dataset, PairwiseDataset, DataCollatorReward
from util.model_utils import get_tokenizer, prepare_merged_model
from util.trl_utils import load_reward_model

if __name__ == "__main__":
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.PT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]
    """
    tokenizer = get_tokenizer(cfg.PT_MODEL)

    _, merged_model = prepare_merged_model(cfg.SFT_MODEL_DIR)
    model = load_reward_model(merged_model)

    val_pairs = create_comparison_dataset(cfg.COMPARISON_DATASET, 10, "test")
    dev_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=cfg.MAX_SUM_LEN)

    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=6, collate_fn=DataCollatorReward())

    correct = 0
    #chosen_list = []
    #reject_list = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
            """
            for x in batch:
                batch[x] = batch[x].cuda()
            """
            print("\n")
            """
            print("--- batch")
            print(batch)
            """
            outputs = model(**batch)
            print("--- chosen_end_scores")
            print(outputs["chosen_end_scores"])
            print("--- rejected_end_scores")
            print(outputs["rejected_end_scores"])
            correct += sum(outputs["chosen_end_scores"] > outputs["rejected_end_scores"])
            #chosen_list.append(outputs["chosen_end_scores"].cpu())
            #reject_list.append(outputs["rejected_end_scores"].cpu())
    print("Total accuracy: ", correct / len(dev_dataset))
