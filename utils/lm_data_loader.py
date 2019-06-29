import os
import torch

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from utils import constant

class LMDataset(Dataset):
    def __init__(self, path, label2id, id2label):
        self.label2id = label2id
        self.id2label = id2label
        self.texts, self.ids = self.read_manifest(path)
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index]
        
    def read_manifest(self, path):
        """Read manifest"""
        texts, ids = [], []
        with open(path, "r") as f:
            for line in f:
                _, text_path = line.replace("\n", "").split(",")
                with open(text_path, "r") as text_file:
                    for l in text_file:
                        texts.append(l.lower().replace("\n", ""))

            for text in texts:
                for char in text:
                    if char not in self.label2id:
                        print(">", char)
                ids.append(list(filter(None, [self.label2id.get(x) for x in list(text)])))
        return texts, ids    

def _collate_fn(batch):
    def func(p):
        return len(p)

    # print(">", batch)
    batch = sorted(batch, key=lambda x: len(x), reverse=True)
    # print(">>", batch)

    max_seq_len = len(max(batch, key=func))
    # print("max_seq_len:", max_seq_len)
    inputs = torch.zeros(len(batch), max_seq_len).long()
    input_sizes = torch.IntTensor(len(batch))

    for i in range(len(batch)):
        sample = batch[i]
        inputs[i][:len(sample)] = torch.IntTensor(sample)

        seq_length = len(sample)
        input_sizes[i] = seq_length

    return inputs, input_sizes

class LMDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(LMDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn