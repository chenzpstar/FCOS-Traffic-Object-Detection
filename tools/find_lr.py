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
from data import BDD100KDataset, Collate, KITTIDataset, VOCDataset
from models import FCOSDetector
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import build_optimizer, mixup, setup_seed


def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bs", default=None, type=int, help="batch size")
    parser.add_argument("--data_folder",
                        default="kitti",
                        type=str,
                        help="dataset folder name")

    return parser.parse_args()


def find_lr(cfg,
            model,
            data_loader,
            optimizer=None,
            scaler=None,
            init_lr=1e-8,
            final_lr=10.0,
            beta=0.98,
            method="exp"):
    lr = init_lr
    optimizer.param_groups[0]["lr"] = lr
    num_steps = len(data_loader) // cfg.acc_steps - 1

    if method == "exp":
        gamma = (final_lr / init_lr)**(1 / num_steps)
    elif method == "linear":
        gamma = (final_lr - init_lr) / num_steps

    lr_rec, loss_rec = [], []
    tmp_loss, avg_loss, best_loss = 0.0, 0.0, 0.0

    for iter, (imgs, labels, boxes) in tqdm(enumerate(data_loader)):
        iter_idx = iter + 1

        imgs = imgs.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)
        boxes = boxes.to(cfg.device, non_blocking=True)

        if cfg.mixup:
            imgs, labels, boxes = mixup(imgs, labels, boxes, cfg.mixup_alpha,
                                        cfg.device)

        if iter_idx % cfg.acc_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        if cfg.use_fp16:
            with autocast():
                cls_loss, reg_loss, ctr_loss = tuple(
                    map(lambda loss: loss / cfg.acc_steps,
                        model(imgs, (labels, boxes))))
                total_loss = cls_loss + reg_loss + ctr_loss

            scaler.scale(total_loss).backward()
            if cfg.clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            if iter_idx % cfg.acc_steps == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            cls_loss, reg_loss, ctr_loss = tuple(
                map(lambda loss: loss / cfg.acc_steps,
                    model(imgs, (labels, boxes))))
            total_loss = cls_loss + reg_loss + ctr_loss

            total_loss.backward()
            if cfg.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            if iter_idx % cfg.acc_steps == 0:
                optimizer.step()

        tmp_loss += total_loss.item()

        if iter_idx % cfg.acc_steps == 0:
            avg_loss = beta * avg_loss + (1.0 - beta) * tmp_loss
            smooth_loss = avg_loss / (1.0 - beta**(iter_idx // cfg.acc_steps))
            tmp_loss = 0.0

            lr_rec.append(lr)
            loss_rec.append(smooth_loss)

            if smooth_loss > 4 * best_loss and iter > 0:
                return lr_rec, loss_rec
            if smooth_loss < best_loss or iter == 0:
                best_loss = smooth_loss

            if method == "exp":
                lr *= gamma
            elif method == "linear":
                lr += gamma
            optimizer.param_groups[0]['lr'] = lr

    return lr_rec, loss_rec


if __name__ == "__main__":
    # 0. config
    setup_seed(seed=0, deterministic=True)

    args = get_args()

    if args.data_folder == "voc":
        from configs.voc_config import cfg
    elif args.data_folder == "kitti":
        from configs.kitti_config import cfg
    elif args.data_folder == "bdd100k":
        from configs.bdd100k_config import cfg

    cfg.train_bs = args.bs if args.bs else cfg.train_bs

    data_dir = os.path.join(cfg.data_root_dir, cfg.data_folder)
    assert os.path.exists(data_dir)

    # 1. data
    if cfg.data_folder == "voc":
        train_set = VOCDataset(
            data_dir,
            year="2007",
            set_name="train",
            transform=cfg.aug_tf,
        )
    elif cfg.data_folder == "kitti":
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
    model = FCOSDetector(cfg).to(cfg.device)
    model.train()
    print("loading model successfully")

    # 3. optimize
    no_decay = ("bias", "norm") if cfg.no_decay else ()
    optimizer = build_optimizer(cfg, model, cfg.optimizer, no_decay)

    scaler = GradScaler() if cfg.use_fp16 else None

    # 4. loop
    lrs, losses = find_lr(cfg, model, train_loader, optimizer, scaler)
    lrs, losses = lrs[10:-5], losses[10:-5]

    min_grad_idx = np.argmin(np.gradient(np.array(losses)))
    suggested_lr = lrs[min_grad_idx]
    print("suggested lr: {:.3e}".format(suggested_lr))

    plt.semilogx(lrs, losses)
    plt.scatter(lrs[min_grad_idx],
                losses[min_grad_idx],
                s=50,
                c="r",
                marker="o")
    plt.xlabel("lr")
    plt.ylabel("loss")
    plt.title("Suggested LR: {:.3e}".format(suggested_lr))
    plt.show()
