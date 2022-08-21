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
# from configs.bdd100k_config import cfg
from configs.kitti_config import cfg
from data import BDD100KDataset, Collate, KITTIDataset
from models import FCOSDetector
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from common import (WarmupLR, build_optimizer, build_scheduler, make_logger,
                    mixup, plot_curve, setup_seed)
from eval import eval_model

# 添加解析参数
parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--lr", default=None, type=float, help="learning rate")
parser.add_argument("--bs", default=None, type=int, help="batch size")
parser.add_argument("--epoch", default=None, type=int, help="num epochs")
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
cfg.init_lr = args.lr if args.lr else cfg.init_lr
cfg.train_bs = args.bs if args.bs else cfg.train_bs
cfg.valid_bs = args.bs if args.bs else cfg.valid_bs
cfg.num_epochs = args.epoch if args.epoch else cfg.num_epochs
cfg.data_folder = args.data_folder if args.data_folder else cfg.data_folder
cfg.ckpt_folder = args.ckpt_folder if args.ckpt_folder else cfg.ckpt_folder


def train_model(cfg,
                model,
                data_loader,
                epoch,
                logger,
                mode="train",
                optimizer=None,
                scheduler=None,
                warmup_scheduler=None,
                scaler=None):
    loss_rec = {"total": [], "cls": [], "reg": [], "ctr": []}
    num_iters = len(data_loader)

    for i, (imgs, labels, boxes) in enumerate(data_loader):
        imgs = imgs.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)
        boxes = boxes.to(cfg.device, non_blocking=True)

        if cfg.mixup:
            imgs, labels, boxes = mixup(imgs, labels, boxes, cfg.mixup_alpha,
                                        cfg.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        if mode == "train":
            if cfg.use_fp16:
                # 1. forward
                with autocast():
                    cls_loss, reg_loss, ctr_loss = tuple(
                        map(lambda loss: loss / cfg.acc_steps,
                            model(imgs, (labels, boxes))))
                    total_loss = cls_loss + reg_loss + ctr_loss

                # 2. backward
                scaler.scale(total_loss).backward()
                if cfg.clip_grad:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(),
                                             cfg.max_grad_norm)

                # 3. update weights
                if (i + 1) % cfg.acc_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # 1. forward
                cls_loss, reg_loss, ctr_loss = tuple(
                    map(lambda loss: loss / cfg.acc_steps,
                        model(imgs, (labels, boxes))))
                total_loss = cls_loss + reg_loss + ctr_loss

                # 2. backward
                total_loss.backward()
                if cfg.clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(),
                                             cfg.max_grad_norm)

                # 3. update weights
                if (i + 1) % cfg.acc_steps == 0:
                    optimizer.step()

            if (i + 1) % cfg.acc_steps == 0:
                # 4. reset grads
                optimizer.zero_grad(set_to_none=True)

                # 5. update lr
                if epoch < cfg.warmup_epochs * cfg.acc_steps:
                    warmup_scheduler.step()
                elif cfg.step_per_iter:
                    scheduler.step()

        elif mode == "valid":
            with torch.no_grad():
                cls_loss, reg_loss, ctr_loss = tuple(
                    map(lambda loss: loss / cfg.acc_steps,
                        model(imgs, (labels, boxes))))
                total_loss = cls_loss + reg_loss + ctr_loss

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cost_time = int((time.time() - start_time) * 1000)

        # 记录训练信息
        if (i + 1) % cfg.log_interval == 0:
            logger.info(
                "{}: epoch: [{:0>2}/{:0>2}], iter: [{:0>3}/{:0>3}], total loss: {:.4f}, cls loss: {:.4f}, reg loss: {:.4f}, ctr loss: {:.4f}, cost time: {} ms"
                .format(mode.title(), epoch + 1, cfg.num_epochs, i + 1,
                        num_iters, total_loss.item(), cls_loss.item(),
                        reg_loss.item(), ctr_loss.item(), cost_time))

        # 记录loss信息
        loss_rec["total"].append(total_loss.item())
        loss_rec["cls"].append(cls_loss.item())
        loss_rec["reg"].append(reg_loss.item())
        loss_rec["ctr"].append(ctr_loss.item())

    return tuple(
        map(lambda loss: np.mean(loss) * cfg.acc_steps, loss_rec.values()))


if __name__ == "__main__":
    # 0. config
    setup_seed(0)

    # 设置路径
    data_dir = os.path.join(cfg.data_root_dir, cfg.data_folder)
    assert os.path.exists(data_dir)

    if cfg.ckpt_folder is not None:
        ckpt_dir = os.path.join(cfg.ckpt_root_dir, cfg.ckpt_folder)
        ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
        assert os.path.exists(ckpt_path)

    # 创建 Logger
    logger, log_dir = make_logger(cfg)

    # 1. data
    # 构建 Dataset
    if cfg.data_folder == "kitti":
        train_set = KITTIDataset(
            data_dir,
            set_name="training",
            mode="train",
            split=True,
            transform=cfg.aug_tf,
        )
        valid_set = KITTIDataset(
            data_dir,
            set_name="training",
            mode="valid",
            split=True,
            transform=cfg.base_tf,
        )
    elif cfg.data_folder == "bdd100k":
        train_set = BDD100KDataset(
            data_dir,
            set_name="train",
            transform=cfg.aug_tf,
        )
        valid_set = BDD100KDataset(
            data_dir,
            set_name="val",
            transform=cfg.base_tf,
        )
    logger.info("train set has {} imgs".format(len(train_set)))
    logger.info("valid set has {} imgs".format(len(valid_set)))

    # 构建 DataLoder
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=cfg.workers,
        collate_fn=Collate(),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=Collate(),
        pin_memory=True,
    )
    logger.info("train loader has {} iters".format(len(train_loader)))
    logger.info("valid loader has {} iters".format(len(valid_loader)))

    # 2. model
    model = FCOSDetector(cfg).to(cfg.device)
    if cfg.ckpt_folder is not None:
        model_weights = torch.load(ckpt_path, map_location=cfg.device)
        model_dict = dict(
            zip(model.state_dict().keys(), model_weights.values()))
        model.load_state_dict(model_dict)
        logger.info("loading checkpoint successfully")
    model.train()
    logger.info("loading model successfully")

    # 3. optimize
    # 构建 Optimizer
    no_decay = ("bias", "norm") if cfg.no_decay else ()
    optimizer = build_optimizer(cfg, model, cfg.optimizer, no_decay)

    # 构建 Scheduler
    num_iters = len(train_loader)
    num_steps = num_iters if cfg.step_per_iter else 1
    scheduler = build_scheduler(cfg, optimizer, cfg.scheduler, num_steps)

    warmup_scheduler = WarmupLR(
        optimizer,
        warmup_factor=cfg.warmup_factor,
        warmup_iters=cfg.warmup_epochs * num_iters,
        warmup_method=cfg.warmup_method,
    ) if cfg.warmup else None

    # 构建 GradScaler
    scaler = GradScaler() if cfg.use_fp16 else None

    # 4. loop
    # 记录配置信息
    logger.info(
        "\ncfg:\n{}\n\noptimizer:\n{}\n\nscheduler:\n{}\n\nmodel:\n{}\n".
        format(cfg, optimizer, scheduler, model))

    total_loss_rec = {"train": [], "valid": []}
    cls_loss_rec = {"train": [], "valid": []}
    reg_loss_rec = {"train": [], "valid": []}
    ctr_loss_rec = {"train": [], "valid": []}

    num_epochs = cfg.num_epochs * cfg.acc_steps
    best_epoch, best_mAP = 0, 0.0

    for epoch in range(num_epochs):
        # 1. train
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total_loss, train_cls_loss, train_reg_loss, train_ctr_loss = train_model(
            cfg, model, train_loader, epoch, logger, "train", optimizer,
            scheduler, warmup_scheduler, scaler)

        # 2. valid
        model.eval()
        valid_total_loss, valid_cls_loss, valid_reg_loss, valid_ctr_loss = train_model(
            cfg, model, valid_loader, epoch, logger, "valid")

        # 记录训练信息
        logger.info(
            "Epoch: [{:0>2}/{:0>2}], lr: {}\n"
            "Train: total loss: {:.4f}, cls loss: {:.4f}, reg loss: {:.4f}, ctr loss: {:.4f}\n"
            "Valid: total loss: {:.4f}, cls loss: {:.4f}, reg loss: {:.4f}, ctr loss: {:.4f}\n"
            .format(epoch + 1, num_epochs, optimizer.param_groups[0]["lr"],
                    train_total_loss, train_cls_loss, train_reg_loss,
                    train_ctr_loss, valid_total_loss, valid_cls_loss,
                    valid_reg_loss, valid_ctr_loss))

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
        plot_curve(plt_x, total_loss_rec, "loss", "total", log_dir)
        plot_curve(plt_x, cls_loss_rec, "loss", "classification", log_dir)
        plot_curve(plt_x, reg_loss_rec, "loss", "regression", log_dir)
        plot_curve(plt_x, ctr_loss_rec, "loss", "centerness", log_dir)

        if (epoch + 1) % cfg.acc_steps == 0:
            # 3. update lr
            if (epoch >= cfg.warmup_epochs * cfg.acc_steps) and (
                    not cfg.step_per_iter):
                scheduler.step()

            # 4. eval
            if epoch >= cfg.milestones[0] * cfg.acc_steps:
                # 评估指标
                metrics = eval_model(model, valid_loader, cfg.num_classes,
                                     cfg.map_iou_thr, cfg.use_07_metric,
                                     cfg.device)

                # 计算 mAP
                mAP = np.mean(metrics["ap"])
                logger.info("mAP: {:.3%}\n".format(mAP))

                # 保存模型
                if mAP > best_mAP:
                    best_epoch, best_mAP = epoch + 1, mAP
                    ckpt_path = os.path.join(log_dir, "checkpoint_best.pth")
                    torch.save(model.state_dict(), ckpt_path)
                    logger.info("saving the best checkpoint successfully\n")

    ckpt_path = os.path.join(log_dir, "checkpoint_last.pth")
    torch.save(model.state_dict(), ckpt_path)
    logger.info("saving the last checkpoint successfully\n")

    logger.info("training model done, best mAP: {:.3%} in epoch: {}".format(
        best_mAP, best_epoch))
