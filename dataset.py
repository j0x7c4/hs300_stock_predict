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
def get_label6(y):
    if y>2:
        return 5
    if y>1:
        return 4
    if y>0:
        return 3
    if y>-1:
        return 2
    if y>-2:
        return 1
    return 0

def get_label3(y):
    if y>1:
        return 2
    if y>-1:
        return 1
    return 0

def get_train_data(opt):
    ratio = opt.ratio
    stock_len = opt.stock_len
    time_step = opt.time_step
    len_index = []
    train_data_path = opt.train_data_path
    df = open(train_data_path)
    data_otrain = pd.read_csv(df)
    data_train = data_otrain.iloc[:, 1:].values
    label_train = data_otrain.iloc[:, -1].values
    data_mean = np.mean(data_train, axis=0)
    data_std = np.std(data_train, axis=0)
    normalized_train_data = (data_train-data_mean)/data_std  # 标准化
    train_x, train_y = [], []   # 训练集x和y定义
    for i in range(len(normalized_train_data) + 1):
        if i % stock_len == 0:
            len_index.append(i)
    for i in range(len(len_index) - 1):
        for k in range(len_index[i], len_index[i + 1] - time_step - 1):
            x = normalized_train_data[k:k + time_step, :6]
            y = label_train[k + time_step]
            train_x.append(x)
            train_y.append(get_label3(y))
    train_x, train_y = np.array(train_x), np.array(train_y)
    train_len = int(len(train_x) * ratio)  # 按照8：2划分训练集和验证集
    train_x_1, train_y_1 = train_x[:train_len], train_y[:train_len]  # 训练集的x和标签
    val_x, val_y = train_x[train_len:], train_y[train_len:]  # 验证集的x和标签
    print(train_x_1[0], train_y_1[0])
    train_set = StockDataset(train_x_1, train_y_1)
    dev_set = StockDataset(val_x, val_y)
    ext_info = {
        "mean": data_mean.tolist(),
        "std": data_std.tolist()
    }
    return train_set, dev_set, ext_info


class StockDataset(Dataset):
    def __init__(self, observations, labels):
        self.observations = observations
        self.labels = labels
        self.size = len(observations)
        self.num_class = max(labels) + 1 

    def __getitem__(self, index):
        item = self.observations[index % self.size]
        item = torch.from_numpy(item)
        label = self.labels[index % self.size]
        return {"x": item, "label": label}

    def __len__(self):
        return self.size


