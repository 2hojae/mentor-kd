import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

from src.utils import json_load


class JsonDataset(TorchDataset):
    """ Load dataset from json file. """
    def __init__(self, filename):
        self.datalist = json_load(filename)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        return self.datalist[i]

    def shuffle(self) -> TorchDataset:
        indices = torch.randperm(len(self))
        dataset = torch.utils.data.Subset(self, indices)
        return dataset


class ArrowDataset(TorchDataset):
    """ Load dataset from arrow table. """
    def __init__(self, filename, in_memory=False):
        self.data = Dataset.from_file(filename, in_memory=in_memory)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
