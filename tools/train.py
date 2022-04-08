# -*- coding: utf-8 -*-
"""
# @file name  : train.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-16
# @brief      : FCOS训练
"""

import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from configs.bdd100k_config import cfg
from data.bdd100k import BDD100KDataset
from data.collate import Collate
from data.kitti import KITTIDataset
from models.fcos import FCOSDetector
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader

from common import make_logger, plot_line, setup_seed

# 添加解析参数
parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--lr", default=None, type=float, help="learning rate")
parser.add_argument("--bs", default=None, type=int, help="train batch size")
parser.add_argument("--max_epoch", default=None, type=int, help="max epoch")
parser.add_argument("--data_folder",
                    default="bdd100k",
                    type=str,
                    help="dataset folder name")
parser.add_argument("--ckpt_folder",
                    default=None,
                    type=str,
                    help="checkpoint folder name")
args = parser.parse_args()

# 修改配置参数
cfg.lr_init = args.lr if args.lr else cfg.lr_init
cfg.train_bs = args.bs if args.bs else cfg.train_bs
cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch

cfg.data_folder = args.data_folder if args.data_folder else cfg.data_folder
cfg.ckpt_folder = (args.ckpt_folder if args.ckpt_folder else cfg.ckpt_folder)

if __name__ == "__main__":
    # 0. config
    setup_seed(0)

    # 设置路径
    if cfg.ckpt_folder is not None:
        ckpt_path = os.path.join(cfg.ckpt_root_dir, cfg.ckpt_folder,
                                 "checkpoint_best.pth")

    data_dir = os.path.join(cfg.data_root_dir, cfg.data_folder)
    assert os.path.exists(data_dir)

    # 创建logger
    out_dir = os.path.join(BASE_DIR, "..", "..", "results")
    logger, log_dir = make_logger(out_dir, cfg)

    # 1. dataset
    # 构建Dataset
    train_set = BDD100KDataset(
        data_dir,
        set_name="train",
        transform=cfg.train_trans,
    )
    valid_set = BDD100KDataset(
        data_dir,
        set_name="val",
        transform=cfg.valid_trans,
    )

    # 构建DataLoder
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=cfg.workers,
        collate_fn=Collate(),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=Collate(),
    )

    # 2. model
    model = FCOSDetector(mode="train", cfg=cfg)
    if cfg.ckpt_folder is not None:
        if os.path.exists(ckpt_path):
            model_weights = torch.load(ckpt_path, map_location="cpu")
            model_weights = {
                k: model_weights[k]
                if k in model_weights else model.state_dict()[k]
                for k in model.state_dict()
            }
            model.load_state_dict(model_weights)
            logger.info("INFO ==> finish loading checkpoint")
        else:
            logger.info(
                "INFO ==> please check your path to checkpoint: {}".format(
                    ckpt_path))
    model.to(cfg.device)
    logger.info("INFO ==> finish loading model")

    # 3. optimize
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr_init,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    if cfg.exp_lr:
        scheduler = ExponentialLR(
            optimizer,
            gamma=cfg.exp_factor,
        )
    else:
        scheduler = MultiStepLR(
            optimizer,
            gamma=cfg.factor,
            milestones=cfg.milestones,
        )

    # 4. loop
    # 记录配置信息
    logger.info(
        "\ncfg:\n{}\n\noptimizer:\n{}\n\nscheduler:\n{}\n\nmodel:\n{}".format(
            cfg, optimizer, scheduler, model))

    loss_rec = {"train": [], "valid": []}

    epochs = cfg.max_epoch
    for epoch in range(epochs):
        # 1. train
        train_loss_sigma = []
        train_loss_mean = 0.0

        model.train()
        for i, data in enumerate(train_loader):
            imgs, labels, boxes = data
            imgs = imgs.to(cfg.device)
            labels = labels.to(cfg.device)
            boxes = boxes.to(cfg.device)

            torch.cuda.synchronize()
            start_time = time.time()

            # 1. forward
            cls_loss, reg_loss, ctr_loss, train_loss = model(
                (imgs, labels, boxes))

            # 2. backward
            optimizer.zero_grad()
            train_loss.mean().backward()

            # 3. update weights
            optimizer.step()

            torch.cuda.synchronize()
            cost_time = int((time.time() - start_time) * 1000)

            # 统计loss
            train_loss_sigma.append(train_loss.mean())

            if (i + 1) % 50 == 0:
                logger.info(
                    "Train: epoch: {:0>3}/{:0>3}, iter: {:0>3}/{:0>3}, cls loss: {:.4f}, reg_loss: {:.4f}, ctr loss: {:.4f}, train loss: {:.4f}, cost time: {} ms"
                    .format(epoch + 1, epochs, i + 1, len(train_loader),
                            cls_loss.mean(), reg_loss.mean(), ctr_loss.mean(),
                            train_loss.mean(), cost_time))

        # 2. valid
        valid_loss_sigma = []
        valid_loss_mean = 0.0

        model.eval()
        for i, data in enumerate(valid_loader):
            imgs, labels, boxes = data
            imgs = imgs.to(cfg.device)
            labels = labels.to(cfg.device)
            boxes = boxes.to(cfg.device)

            torch.cuda.synchronize()
            start_time = time.time()

            # forward
            with torch.no_grad():
                cls_loss, reg_loss, ctr_loss, valid_loss = model(
                    (imgs, labels, boxes))

            torch.cuda.synchronize()
            cost_time = int((time.time() - start_time) * 1000)

            # 统计loss
            valid_loss_sigma.append(valid_loss.mean())

            if (i + 1) % 50 == 0:
                logger.info(
                    "Valid: epoch: {:0>3}/{:0>3}, iter: {:0>3}/{:0>3}, cls loss: {:.4f}, reg_loss: {:.4f}, ctr loss: {:.4f}, valid loss: {:.4f}, cost time: {} ms"
                    .format(epoch + 1, epochs, i + 1, len(valid_loader),
                            cls_loss.mean(), reg_loss.mean(), ctr_loss.mean(),
                            valid_loss.mean(), cost_time))

        # 3. update lr
        scheduler.step()

        train_loss_mean = np.mean(train_loss_sigma)
        valid_loss_mean = np.mean(valid_loss_sigma)

        logger.info(
            "epoch: {:0>3}/{:0>3}, lr: {}, train loss: {:.4f}, valid loss: {:.4f}"
            .format(epoch + 1, epochs, optimizer.param_groups[0]["lr"],
                    train_loss_mean, valid_loss_mean))

        # 记录loss信息
        loss_rec["train"].append(train_loss_mean)
        loss_rec["valid"].append(valid_loss_mean)

        # 保存loss曲线
        plt_x = np.arange(1, epoch + 2)
        plot_line(
            plt_x,
            loss_rec["train"],
            plt_x,
            loss_rec["valid"],
            mode="loss",
            out_dir=log_dir,
        )

        # 保存模型
        torch.save(
            model.state_dict(),
            os.path.join(log_dir, "checkpoint_{}.pth".format(epoch + 1)))

    logger.info("INFO ==> finish training model")
