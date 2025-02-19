from torch.utils.data import Dataset
import pandas as pd
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# external storage for entire kaggle dataset
DATA_PATH = "D:/analytics_projects/summarization/cnn_dailymail"

# TODO: implement load_data and corresponding dataloaders if project eventually trains
class SummarizationDataset(Dataset):
    def __init__(self, article, summary, model_name, max_length=512):
        self.article = article
        self.summary = summary
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.article)

    def __getitem__(self, idx):
        src_text = self.article[idx]
        tgt_text = self.summary[idx]

        src_enc = self.tokenizer(
            src_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        tgt_enc = self.tokenizer(
            tgt_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": src_enc["input_ids"].squeeze(0),
            "attention_mask": src_enc["attention_mask"].squeeze(0),
            "labels": tgt_enc["input_ids"].squeeze(0),
        }


def create_dataloader(df, args):

    src = df[args.src_col]
    tgt = df[args.tgt_col]

    dataset = SummarizationDataset(src, tgt, args.model_name)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataloader


def load_data(args, file, dataloader=True):

    if file == "train":
        df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))

    elif file == "test":
        df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

    elif file == "valid":
        df = pd.read_csv(os.path.join(DATA_PATH, "validation.csv"))

    else:
        print(f"mode {file} not supported. Please use either: train, test, or valid.")

    if dataloader:
        return create_dataloader(df, args)
    else:
        return df


def load_df():
    df = pd.read_csv("data/zero_shot_df.csv")
    return df
