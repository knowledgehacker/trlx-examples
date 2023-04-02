import torch
from rm.reward_model import OPTRewardModel
from tqdm import tqdm
#from transformers import AutoTokenizer

import config as cfg
from rm.comparison_dataset import create_comparison_dataset, PairwiseDataset, DataCollatorReward
from model_loader import get_tokenizer


if __name__ == "__main__":
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.PT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]
    """
    tokenizer = get_tokenizer(cfg.PT_MODEL)
    PAD_ID = tokenizer.pad_token_id
    print("PAD_ID: %s" % PAD_ID)

    # TODO: how to save rm model to cfg.RM_CKPT_PATH?
    model = OPTRewardModel(cfg.SFT_CKPT_DIR)
    model.load_state_dict(torch.load(cfg.RM_CKPT_PATH))
    max_length = 550
    val_pairs = create_comparison_dataset(cfg.SUMMARIZATION_DATASET, 10, "test")
    dev_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    from torch.utils.data import DataLoader

    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=6, collate_fn=DataCollatorReward())
    model.cuda()
    model.eval()
    model.half()
    correct = 0
    chosen_list = []
    reject_list = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
            for x in batch:
                batch[x] = batch[x].cuda()
            outputs = model(**batch)
            correct += sum(outputs["chosen_end_scores"] > outputs["rejected_end_scores"])
            chosen_list.append(outputs["chosen_end_scores"].cpu())
            reject_list.append(outputs["rejected_end_scores"].cpu())
    print("Total accuracy: ", correct / len(dev_dataset))
