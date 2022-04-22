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
from apex import amp
# from configs.bdd100k_config import cfg
from configs.kitti_config import cfg
from data import BDD100KDataset, Collate, KITTIDataset
from models.fcos import FCOSDetector
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR,
                                      MultiStepLR)
from torch.utils.data import DataLoader

from common import WarmupLR, make_logger, plot_curve, setup_seed

# 添加解析参数
parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--lr", default=None, type=float, help="learning rate")
parser.add_argument("--bs", default=None, type=int, help="train batch size")
parser.add_argument("--max_epoch", default=None, type=int, help="max epoch")
parser.add_argument("--data_folder",
                    default="kitti",
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
cfg.ckpt_folder = args.ckpt_folder if args.ckpt_folder else cfg.ckpt_folder


def train_model(cfg,
                data_loader,
                model,
                epoch,
                logger,
                optimizer=None,
                scheduler=None,
                mode="train"):
    total_loss_sigma = []
    cls_loss_sigma = []
    reg_loss_sigma = []
    ctr_loss_sigma = []

    for i, (imgs, labels, boxes) in enumerate(data_loader):
        imgs = imgs.to(cfg.device)
        labels = labels.to(cfg.device)
        boxes = boxes.to(cfg.device)

        torch.cuda.synchronize()
        start_time = time.time()

        if mode == "train":
            # 1. forward
            total_loss, cls_loss, reg_loss, ctr_loss = model(
                (imgs, labels, boxes))

            # 2. backward
            optimizer.zero_grad()

            if cfg.use_fp16:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), 5)
            else:
                total_loss.backward()
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            # 3. update weights
            optimizer.step()

            # 4. update lr
            if cfg.warmup:
                scheduler.step()

        elif mode == "valid":
            # 1. forward
            with torch.no_grad():
                total_loss, cls_loss, reg_loss, ctr_loss = model(
                    (imgs, labels, boxes))

        torch.cuda.synchronize()
        cost_time = int((time.time() - start_time) * 1000)

        # 统计loss
        total_loss_sigma.append(total_loss.item())
        cls_loss_sigma.append(cls_loss.item())
        reg_loss_sigma.append(reg_loss.item())
        ctr_loss_sigma.append(ctr_loss.item())

        if (i + 1) % cfg.log_interval == 0:
            logger.info(
                "{}: epoch: [{:0>2}/{:0>2}], iter: [{:0>3}/{:0>3}], total loss: {:.4f}, cls loss: {:.4f}, reg loss: {:.4f}, ctr loss: {:.4f}, cost time: {} ms"
                .format(mode.title(), epoch + 1, cfg.max_epoch, i + 1,
                        len(data_loader), total_loss.item(), cls_loss.item(),
                        reg_loss.item(), ctr_loss.item(), cost_time))

    return (
        np.mean(total_loss_sigma),
        np.mean(cls_loss_sigma),
        np.mean(reg_loss_sigma),
        np.mean(ctr_loss_sigma),
    )


if __name__ == "__main__":
    # 0. config
    setup_seed(0)

    # 设置路径
    if cfg.ckpt_folder is not None:
        ckpt_dir = os.path.join(cfg.ckpt_root_dir, cfg.ckpt_folder)
        ckpt_path = os.path.join(ckpt_dir, "checkpoint_12.pth")
        assert os.path.exists(ckpt_path)

    data_dir = os.path.join(cfg.data_root_dir, cfg.data_folder)
    assert os.path.exists(data_dir)

    # 创建logger
    logger, log_dir = make_logger(cfg)

    # 1. dataset
    # 构建Dataset
    if cfg.data_folder == "kitti":
        train_set = KITTIDataset(
            data_dir,
            set_name="training",
            mode="train",
            split=True,
            transform=cfg.aug_trans,
        )
        valid_set = KITTIDataset(
            data_dir,
            set_name="training",
            mode="valid",
            split=True,
            transform=cfg.base_trans,
        )
    elif cfg.data_folder == "bdd100k":
        train_set = BDD100KDataset(
            data_dir,
            set_name="train",
            transform=cfg.aug_trans,
        )
        valid_set = BDD100KDataset(
            data_dir,
            set_name="val",
            transform=cfg.base_trans,
        )
    logger.info("train set has {} imgs".format(len(train_set)))
    logger.info("valid set has {} imgs".format(len(valid_set)))

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
    logger.info("train loader has {} iters".format(len(train_loader)))
    logger.info("valid loader has {} iters".format(len(valid_loader)))

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
            logger.info("finish loading checkpoint")
        else:
            logger.info(
                "please check your path to checkpoint: {}".format(ckpt_path))
    model.to(cfg.device)
    logger.info("finish loading model")

    # 3. optimize
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr_init,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    if cfg.use_fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if cfg.exp_lr:
        scheduler = ExponentialLR(
            optimizer,
            gamma=cfg.exp_factor,
        )
    elif cfg.cos_lr:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.max_epoch - 1,
            eta_min=cfg.lr_final,
        )
    else:
        scheduler = MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.factor,
        )

    if cfg.warmup:
        warmup_scheduler = WarmupLR(
            optimizer,
            warmup_factor=cfg.warmup_factor,
            warmup_iters=len(train_loader),
        )
    else:
        warmup_scheduler = None

    # 4. loop
    # 记录配置信息
    logger.info(
        "\ncfg:\n{}\n\noptimizer:\n{}\n\nscheduler:\n{}\n\nmodel:\n{}\n".
        format(cfg, optimizer, scheduler, model))

    total_loss_rec = {"train": [], "valid": []}
    cls_loss_rec = {"train": [], "valid": []}
    reg_loss_rec = {"train": [], "valid": []}
    ctr_loss_rec = {"train": [], "valid": []}

    epochs = cfg.max_epoch
    for epoch in range(epochs):
        # 1. train
        model.train()
        train_total_loss, train_cls_loss, train_reg_loss, train_ctr_loss = train_model(
            cfg,
            train_loader,
            model,
            epoch,
            logger,
            optimizer=optimizer,
            scheduler=warmup_scheduler,
            mode="train")

        # 2. valid
        model.eval()
        valid_total_loss, valid_cls_loss, valid_reg_loss, valid_ctr_loss = train_model(
            cfg, valid_loader, model, epoch, logger, mode="valid")

        # 记录训练信息
        logger.info(
            "Epoch: [{:0>2}/{:0>2}], lr: {}\n"
            "Train: total loss: {:.4f}, cls loss: {:.4f}, reg loss: {:.4f}, ctr loss: {:.4f}\n"
            "Valid: total loss: {:.4f}, cls loss: {:.4f}, reg loss: {:.4f}, ctr loss: {:.4f}\n"
            .format(epoch + 1, epochs, optimizer.param_groups[0]["lr"],
                    train_total_loss, train_cls_loss, train_reg_loss,
                    train_ctr_loss, valid_total_loss, valid_cls_loss,
                    valid_reg_loss, valid_ctr_loss))

        # 3. update lr
        scheduler.step()
        cfg.warmup = False

        # 记录loss信息
        total_loss_rec["train"].append(train_total_loss)
        total_loss_rec["valid"].append(valid_total_loss)
        cls_loss_rec["train"].append(train_cls_loss)
        cls_loss_rec["valid"].append(valid_cls_loss)
        reg_loss_rec["train"].append(train_reg_loss)
        reg_loss_rec["valid"].append(valid_reg_loss)
        ctr_loss_rec["train"].append(train_ctr_loss)
        ctr_loss_rec["valid"].append(valid_ctr_loss)

        # 绘制loss曲线
        plt_x = np.arange(1, epoch + 2)
        plot_curve(
            plt_x,
            total_loss_rec["train"],
            plt_x,
            total_loss_rec["valid"],
            mode="loss",
            kind="total",
            out_dir=log_dir,
        )
        plot_curve(
            plt_x,
            cls_loss_rec["train"],
            plt_x,
            cls_loss_rec["valid"],
            mode="loss",
            kind="classification",
            out_dir=log_dir,
        )
        plot_curve(
            plt_x,
            reg_loss_rec["train"],
            plt_x,
            reg_loss_rec["valid"],
            mode="loss",
            kind="regression",
            out_dir=log_dir,
        )
        plot_curve(
            plt_x,
            ctr_loss_rec["train"],
            plt_x,
            ctr_loss_rec["valid"],
            mode="loss",
            kind="centerness",
            out_dir=log_dir,
        )

        # 保存模型
        if epoch >= cfg.milestones[0]:
            ckpt_path = os.path.join(log_dir,
                                     "checkpoint_{}.pth".format(epoch + 1))
            torch.save(model.state_dict(), ckpt_path)

    logger.info("finish training model")
