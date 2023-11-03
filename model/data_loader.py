import random
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class EMGDataset(Dataset):
    def __init__(self, filename, transform=None):
        emg_df = pd.read_csv(filename, header=None)
        feat = emg_df.iloc[:, :-1].values
        label = emg_df.iloc[:, -1].values

        self.x = torch.from_numpy(feat).float()
        self.y = torch.from_numpy(label).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx].reshape((1, 500, 12)).transpose(1, 2), self.y[idx]


def fetch_dataloader(type, filename, params):
    emg_dataset = EMGDataset(filename)
    shuffle_flag = True if type == "Train" else False
    emg_dataloader = DataLoader(
        emg_dataset,
        batch_size=params.batch_size,
        shuffle=shuffle_flag,
        pin_memory=params.cuda,
    )

    return emg_dataloader



