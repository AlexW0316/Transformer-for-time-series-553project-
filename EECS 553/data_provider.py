from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch


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
horizon = 1

all_dataset = None
for i in range(window_size, n_dates):
    window_X = X[i - window_size:i, :, :]  # [window_size, n_stocks, n_features (=2)]
    window_y = y[i:i + horizon, :]  # [horizon, n_stocks]
    start_date = dates[i]
    cur_dataset = TimeSeriesDataset(window_X, window_y, start_date, n_stocks)
    all_dataset = all_dataset + cur_dataset if all_dataset is not None else cur_dataset


n_total = len(all_dataset)
train_size = int(0.7 * n_total)
valid_size = int(0.2 * n_total)
test_size = n_total - train_size - valid_size
dataloader_train, dataloader_valid, dataloader_test =\
    torch.utils.data.random_split(all_dataset, [train_size, valid_size, test_size])
# # Returns [batch_size, window_size, n_features] and [batch_size, horizon] from a random sector

batch_size = 32

