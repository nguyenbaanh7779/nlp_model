import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.context = self.data["document"]
        self.summaries = self.data["summary"]

    def __len__(self):
        return len(self.context)

    def __getitem__(self, index):
        context = self.context[index]
        summary = self.summaries[index]

        source = self.tokenizer.batch_encode_plus(
            [context],
            max_length=self.source_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [summary],
            max_length=self.summ_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        # target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }