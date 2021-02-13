# coding:utf-8
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import TransformerModel
from dataset import get_train_data
import torch
from functions import (
    move_to_gpu
)

def train(train_data, eval_data=None):
    print("start training task")
    try:
        dataloader = DataLoader(
            train_data,
            batch_size=128,
            shuffle=False,
            num_workers=4
        )
        print("create training dataloader")
    except Exception as e:
        print("fail to create dataloader", e)

    model = move_to_gpu(TransformerModel(6, 512, 8, 6, 3))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model.optimizer,
            milestones=[30, 80], gamma=0.1)

    global_steps = 0
    best_auc = 0

    for epoch in tqdm(list(range(3)), desc='epoch'):
        for step, batch in enumerate(dataloader):
            print(batch)
            global_steps += 1
            model.train(batch)
        lr_scheduler.step()
    return


if __name__ == "__main__":
    train_set, dev_set = get_train_data()
    train(train_set)
