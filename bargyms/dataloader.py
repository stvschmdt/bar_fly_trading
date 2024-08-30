# file to create pytorch dataloader for the dataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torchvision import transforms

class DailyData(Dataset):
    def __init__(self,
                 df,
                 symbols,
                 state_days,
                 n_days,
                 transform=None,
                 target_transform=None):
        self.df = df
        self.symbols = symbols
        self.state_days = state_days
        self.n_days = n_days
        self.transform = transform
        self.target_transform = target_transform
        # filter data to only symbols in list
        self.df = self.df[self.df['symbol'].isin(self.symbols)]

    def __len__(self):
        return len(self.df) - (self.state_days + self.n_days + 1)

    def __getitem__(self, idx):
        # access window of dataframe date data, ensure no wrap around
        if idx + self.state_days > len(self.df):
            idx = len(self.df) - (self.state_days + self.n_days+1)
        window = self.df.iloc[idx:idx+self.state_days]
        # get label, n days out from window
        label = self.df.iloc[idx+self.state_days+self.n_days]
        if self.transform is not None:
            pass
        if self.target_transform:
            label = self.target_transform(label)

        label = torch.from_numpy(np.asarray(label))

        return window, label
