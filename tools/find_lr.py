# -*- coding: utf-8 -*-
"""
# @file name  : find_lr.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-06
# @brief      : 寻找学习率
# @reference  : https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
# from configs.bdd100k_config import cfg
from configs.kitti_config import cfg
from data import BDD100KDataset, Collate, KITTIDataset
from models import FCOSDetector
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import setup_seed

# 添加解析参数
parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--bs", default=None, type=int, help="batch size")
parser.add_argument("--data_folder",
                    default="kitti",
                    type=str,
                    help="dataset folder name")
args = parser.parse_args()

# 修改配置参数
cfg.train_bs = args.bs if args.bs else cfg.train_bs
cfg.data_folder = args.data_folder if args.data_folder else cfg.data_folder


def find_lr(cfg,
            model,
            data_loader,
            optimizer=None,
            scaler=None,
            init_lr=1e-8,
            final_lr=10.0,
            beta=0.98,
            mode="exp"):
    lr = init_lr
    optimizer.param_groups[0]["lr"] = lr
    num_iters = len(data_loader) - 1

    if mode == "exp":
        gamma = (final_lr / init_lr)**(1 / num_iters)
    elif mode == "linear":
        gamma = (final_lr - init_lr) / num_iters

    lr_rec, loss_rec = [], []
    avg_loss, best_loss = 0.0, 0.0

    for i, (imgs, labels, boxes) in tqdm(enumerate(data_loader)):
        imgs = imgs.to(cfg.device)
        labels = labels.to(cfg.device)
        boxes = boxes.to(cfg.device)

        optimizer.zero_grad()

        if cfg.use_fp16:
            with autocast():
                losses = model(imgs, (labels, boxes))
            scaler.scale(losses[0]).backward()
            if cfg.clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses = model(imgs, (labels, boxes))
            losses[0].backward()
            if cfg.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

        avg_loss = beta * avg_loss + (1 - beta) * losses[0].item()
        smooth_loss = avg_loss / (1 - beta**(i + 1))

        lr_rec.append(lr)
        loss_rec.append(smooth_loss)

        if smooth_loss > 4 * best_loss and i > 0:
            return lr_rec, loss_rec
        if smooth_loss < best_loss or i == 0:
            best_loss = smooth_loss

        if mode == "exp":
            lr *= gamma
        elif mode == "linear":
            lr += gamma
        optimizer.param_groups[0]['lr'] = lr

    return lr_rec, loss_rec


if __name__ == "__main__":
    # 0. config
    setup_seed(0)

    # 设置路径
    data_dir = os.path.join(cfg.data_root_dir, cfg.data_folder)
    assert os.path.exists(data_dir)

    # 1. dataset
    # 构建Dataset
    if cfg.data_folder == "kitti":
        train_set = KITTIDataset(
            data_dir,
            set_name="training",
            mode="train",
            split=True,
            transform=cfg.aug_tf,
        )
    elif cfg.data_folder == "bdd100k":
        train_set = BDD100KDataset(
            data_dir,
            set_name="train",
            transform=cfg.aug_tf,
        )
    print("train set has {} imgs".format(len(train_set)))

    # 构建DataLoder
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=cfg.workers,
        collate_fn=Collate(),
        pin_memory=True,
    )
    print("train loader has {} iters".format(len(train_loader)))

    # 2. model
    model = FCOSDetector(cfg)
    model.to(cfg.device)
    model.train()
    print("loading model successfully")

    # 3. optimize
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.init_lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    scaler = GradScaler() if cfg.use_fp16 else None

    # 4. loop
    lr_rec, loss_rec = find_lr(
        cfg,
        model,
        train_loader,
        optimizer=optimizer,
        scaler=scaler,
        init_lr=1e-8,
        final_lr=10,
        beta=0.98,
        mode="exp",
    )

    min_grad_idx = np.argmin(np.gradient(np.array(loss_rec)))
    suggest_lr = lr_rec[min_grad_idx]

    plt.plot(lr_rec[10:-5], loss_rec[10:-5])
    plt.scatter(lr_rec[min_grad_idx],
                loss_rec[min_grad_idx],
                s=50,
                c="r",
                marker="o")
    plt.xscale("log")
    plt.xlabel("lr")
    plt.ylabel("loss")
    plt.title("suggest lr: {:.3e}".format(suggest_lr))
    plt.show()
