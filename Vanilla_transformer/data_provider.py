from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, date, n_stocks):
        self.X = X  # [window_size, n_stocks, n_features (=2)]
        self.y = y  # [horizon, n_stocks]
        self.date = date
        self.n_stocks = n_stocks

    def __len__(self):
        return self.n_stocks

    def __getitem__(self, idx):
        return self.X[:, idx, :], self.y[:, idx]  # [window_size, n_features], [horizon]


class TimeSeriesDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        super(TimeSeriesDataLoader, self).__init__(dataset, batch_size, shuffle=shuffle)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle


price_df = pd.read_csv('price_data.csv', index_col=0)
vol_df = pd.read_csv('vol_data.csv', index_col=0)

n_stocks = len(price_df.columns)
dates = pd.to_datetime(price_df.index).strftime("%Y%m%d").astype(int)
n_dates = len(dates)

X = torch.tensor(
    np.concatenate(
        [price_df.values.reshape(-1, n_stocks, 1), vol_df.values.reshape(-1, n_stocks, 1)],
        axis=-1
    )
)  # [n_dates, n_stocks, n_features (=2)]

y = torch.tensor(price_df.values)  # [n_dates, n_stocks]

window_size = 96

all_dataset = []
for i in range(window_size, n_dates - (window_size + 20)):
    window_X = X[i - window_size:i, :, :]  # [window_size, n_stocks, n_features (=2)]
    window_y = y[i + 20: i + window_size + 20, :]  # [window_size, n_stocks]
    start_date = dates[i]
    cur_dataset = TimeSeriesDataset(window_X, window_y, start_date, n_stocks)
    all_dataset.append(cur_dataset)
all_dataset = ConcatDataset(all_dataset)


n_total = len(all_dataset)
print(n_total)
train_size = int(0.7 * n_total)
valid_size = n_total - train_size
dataset_train, dataset_valid = \
    torch.utils.data.random_split(all_dataset, [train_size, valid_size])

batch_size = 32
dataloader_train, dataloader_valid = \
    DataLoader(dataset_train, batch_size=batch_size, shuffle=True), \
        DataLoader(dataset_valid, batch_size=batch_size, shuffle=True), \
    # Returns [batch_size, window_size, n_features] and [batch_size, horizon] from a random sector



# apple_stock


price_df = pd.read_csv('apple_stock.csv', index_col=0)

n_stocks = len(price_df.columns)
dates = pd.to_datetime(price_df.index).strftime("%Y%m%d").astype(int)
n_dates = len(dates)

print()

X = torch.tensor(price_df.values.reshape(-1, n_stocks, 1))  # [n_dates, n_stocks, n_features (=2)]

y = torch.tensor(price_df.values)  # [n_dates, n_stocks]

window_size = 96

all_dataset = []
for i in range(window_size, n_dates - (window_size + 20)):
    window_X = X[i - window_size:i, :, :]  # [window_size, n_stocks, n_features (=2)]
    window_y = y[i + 20: i + window_size + 20, :]  # [window_size, n_stocks]
    start_date = dates[i]
    cur_dataset = TimeSeriesDataset(window_X, window_y, start_date, n_stocks)
    all_dataset.append(cur_dataset)
all_dataset = ConcatDataset(all_dataset)


n_total = len(all_dataset)
print(n_total)
train_size = int(0.7 * n_total)
valid_size = n_total - train_size
dataset_train, dataset_valid = \
    torch.utils.data.random_split(all_dataset, [train_size, valid_size])

batch_size = 32
dataloader_train, dataloader_valid = \
    DataLoader(dataset_train, batch_size=batch_size, shuffle=True), \
        DataLoader(dataset_valid, batch_size=batch_size, shuffle=True), \
    # Returns [batch_size, window_size, n_features] and [batch_size, horizon] from a random sector
