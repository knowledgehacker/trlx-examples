import evaluate

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from trl import AutoModelForCausalLMWithValueHead

import config as cfg
from util.model_utils import get_tokenizer, prepare_merged_model
from util.trl_utils import build_dataset, collator#, load_reward_model, get_scores
from util.token_utils import encode, decode


# tokenizer
tokenizer = get_tokenizer(cfg.PT_MODEL)

# model
ppo_model, merged_model = prepare_merged_model(cfg.PPO_MODEL_DIR)
model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_model)
#model.cuda()
model.eval()

#reward_model = load_reward_model(merged_model)
#reward_model.to(cfg.device)

generation_kwargs = {
            # "do_sample": True,  # sample a response from top k probabilities
            "max_new_tokens": cfg.MAX_NEW_TOKENS,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }


def inference(model, test_dataset):
    rouge = evaluate.load("rouge")

    pred_list = []

    prompt_list = []
    summarize_list = []

    count = 0

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
    )
    for epoch, batch in tqdm(enumerate(data_loader)):
        for input_ids, attention_mask in zip(batch["input_ids"], batch["attention_mask"]):
            with torch.cuda.amp.autocast():
                response_with_prompt = model.generate(input_ids=input_ids.unsqueeze(dim=0), attention_mask=attention_mask.unsqueeze(dim=0), **generation_kwargs)
            # strip prompt tokens, leaving response tokens only
            response = response_with_prompt.squeeze()[-generation_kwargs["max_new_tokens"]:]
            response = decode(tokenizer, response.squeeze())
            print("--- response")
            print(response)
            pred_list.append(response)

        prompt = batch["query"]
        print("--- prompt")
        print(prompt)
        prompt_list.append(prompt)
        summary = batch["label"]
        print("--- summary")
        print(summary)
        summarize_list.append(summary)

        if count % 10 == 0:
            result = rouge.compute(predictions=pred_list, references=summarize_list)
            print(result)
        count += 1
    df = pd.DataFrame.from_dict({"post": prompt_list, "truth": summarize_list, "pred": pred_list})
    result = rouge.compute(predictions=pred_list, references=summarize_list)
    print(result)
    return df


def write_result(df_result):
    #sup_pred = df_result["pred"].values
    #truth = df_result["truth"].values

    #scores_pred = []
    #scores_truth = []
    preds_list = []
    truth_list = []
    post_list = []
    batch_size = 16
    for i in range(0, len(df_result), batch_size):
        predicts = df_result["pred"].values[i: i + batch_size]
        labels = df_result["truth"].values[i: i + batch_size]
        posts = df_result["post"].values[i: i + batch_size]
        #data_pred = [posts[i] + predicts[i] for i in range(len(predicts))]
        #data_truth = [posts[i] + labels[i] for i in range(len(labels))]
        preds_list.extend(list(predicts))
        truth_list.extend(list(labels))
        post_list.extend(list(posts))
        # scores_pred.extend(list(get_scores(tokenizer, reward_model, data_pred).cpu().numpy()))
        # scores_truth.extend(list(get_scores(tokenizer, reward_model, data_truth).cpu().numpy()))

    df = pd.DataFrame.from_dict(
        {
            "pred": preds_list,
            "truth": truth_list,
            "post": post_list,
            # "score_pred": scores_pred,
            # "score_truth": scores_truth,
        }
    )
    df.to_csv("ppo_with_reward_scores.csv", index=False)
    # print("Reward score pred: ", df.score_pred.values.mean())
    # print("Reward score truth: ", df.score_truth.values.mean())


if __name__ == "__main__":
    max_input_len = cfg.MAX_SUM_LEN - cfg.MAX_NEW_TOKENS
    test_dataset, test_prompt_summary_dict = build_dataset(tokenizer, cfg.SUMMARIZATION_DATASET, 2, "test", max_input_len)

    df_result = inference(model, test_dataset)
    write_result(df_result)
