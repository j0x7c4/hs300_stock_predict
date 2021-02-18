# coding:utf-8
import os
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

# 该系列代码所要求的股票文件名称必须是股票代码+csv的格式，如000001.csv
# --------------------------训练集数据的处理--------------------- #
def get_train_data(opt):
    ratio = opt.ratio
    stock_len = opt.stock_len
    time_step = opt.time_step
    len_index = []
    train_data_path = opt.train_data_path
    df = open(train_data_path)
    data_otrain = pd.read_csv(df)
    data_train = data_otrain.iloc[:, 1:].values
    # label_train = data_otrain['close'] / data_otrain['open'] - 1
    # label_train = label_train.values
    # print(label_train)
    label_train = data_otrain.iloc[:, -1].values
    normalized_train_data = (data_train-np.mean(data_train, axis=0))/np.std(data_train, axis=0)  # 标准化
    # normalized_train_data = data_train  # 标准化
    train_x, train_y = [], []   # 训练集x和y定义
    for i in range(len(normalized_train_data) + 1):
        if i % stock_len == 0:
            len_index.append(i)
    for i in range(len(len_index) - 1):
        for k in range(len_index[i], len_index[i + 1] - time_step - 1):
            x = normalized_train_data[k:k + time_step, :6]
            y = label_train[k + time_step]
            temp_data = 0
            # onehot编码
            if y > 2:
                temp_data = 5
            elif 1 < y <= 2:
                temp_data = 4
            elif 0 < y <= 1:
                temp_data = 3
            elif -1 < y <= 0:
                temp_data = 2
            elif -2 < y <= -1:
                temp_data = 1
            train_x.append(x)
            train_y.append(temp_data)
    train_x, train_y = np.array(train_x), np.array(train_y)
    train_len = int(len(train_x) * ratio)  # 按照8：2划分训练集和验证集
    train_x_1, train_y_1 = train_x[:train_len], train_y[:train_len]  # 训练集的x和标签
    val_x, val_y = train_x[train_len:], train_y[train_len:]  # 验证集的x和标签
    print(train_x_1[0], train_y_1[0])
    train_set = StockDataset(train_x_1, train_y_1)
    dev_set = StockDataset(val_x, val_y)
    return train_set, dev_set


class StockDataset(Dataset):
    def __init__(self, observations, labels):
        self.observations = observations
        self.labels = labels
        self.size = len(observations)

    def __getitem__(self, index):
        item = self.observations[index % self.size]
        item = torch.from_numpy(item)
        label = self.labels[index % self.size]
        return {"x": item, "label": label}

    def __len__(self):
        return self.size


