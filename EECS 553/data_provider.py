from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, date, n_stocks):
        self.X = X  # [n_dates, n_stocks, n_features (=2)]
        self.y = y  # [n_dates, n_stocks]
        self.date = date
        self.n_stocks = n_stocks

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[:,idx,:], self.y[idx]


class TimeSeriesDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        super(TimeSeriesDataLoader, self).__init__(dataset, batch_size, shuffle=shuffle)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

