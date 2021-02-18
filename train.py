# coding:utf-8
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import TransformerModel
from dataset import get_train_data
import torch
from config import get_arguments
from functions import (
    move_to_gpu
)
import numpy as np
from metrics import Metrics
import os
import logging
import torch.nn as nn
from torchsummary import summary
# logging.basicConfig(format='%(asctime)s %(message)s',
#        filename="./logs/train.log", filemode='w', level="DEBUG")
logging.basicConfig(format='%(asctime)s %(message)s',level="DEBUG")
logger = logging.getLogger(__file__)

def eval(opt, model, eval_data):
    logger.info("start eval task")
    try:
        dataloader = DataLoader(
            eval_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=4
        )
        logger.info("create eval dataloader")
    except Exception as e:
        logger.error("fail to create dataloader", e)

    global_steps = 0
    labels, preds, scores = [], [], []
    losses = []
    for step, batch in enumerate(dataloader):
        global_steps += 1
        result = model.predict(batch)
        idx = torch.argmax(result, dim=1)
        score = torch.index_select(result, 1, idx)
        label = batch['label']
        preds += idx.tolist()
        labels += label.tolist()
        scores += score.tolist()
        if model.cuda:
            label = move_to_gpu(label)
        loss = model.loss_fn(result, label)
        losses.append(loss.item())
    mAP, info = Metrics.mAP(preds, labels)
    info.update({"mAP": mAP, "eval_loss": np.average(losses)})
    return mAP, info

def train(opt, train_data, eval_data=None):
    logger.info("start training task")
    dim_input = 6
    dim_emb = 64
    num_class = train_data.num_class
    transformer_nhead = 2
    transformer_nlayers = 1
    model = TransformerModel(dim_input, dim_emb, transformer_nhead,
        num_class,
        transformer_nlayers)
    if model.cuda:
        model = move_to_gpu(model)
    summary(model, train_data[0]['x'].shape)
    try:
        dataloader = DataLoader(
            train_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=4
        )
        logger.info("create training dataloader")
    except Exception as e:
        logger.error("fail to create dataloader", e)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model.optimizer,
            milestones=[5, 10], gamma=0.1)

    model_path = os.path.join(opt.model_dir,opt.model_name+".pth")
    global_steps = 0
    best = 0
    for epoch in tqdm(list(range(opt.epoch)), desc='epoch'):
        for step, batch in enumerate(dataloader):
            global_steps += 1
            metrics = model.train(batch)
            if global_steps % opt.log_steps == 0:
                logger.debug(f"global steps={global_steps},{metrics}")
            if global_steps % opt.save_steps == 0:
                val_metrics, eval_result = eval(opt, model, eval_data)
                logger.info(f"global steps={global_steps}, current={val_metrics}, best={best}, result={eval_result}")
                if val_metrics > best:
                    best = val_metrics
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"global steps={global_steps}, save model:{model_path}")
        lr_scheduler.step()


if __name__ == "__main__":
    parser = get_arguments()
    opt = parser.parse_args()
    train_set, dev_set = get_train_data(opt)
    train(opt, train_set, dev_set)
