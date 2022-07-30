# -*- coding: utf-8 -*-
"""
# @file name  : common.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-16
# @brief      : 通用函数
"""

import logging
import math
import os
import random
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR,
                                      MultiStepLR, _LRScheduler)


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def mixup(imgs, labels, boxes, alpha=1.5, device="cpu"):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(imgs.shape[0]).to(device, non_blocking=True)

    mixed_imgs = lam * imgs + (1.0 - lam) * imgs[idx, :]
    concat_labels = torch.cat((labels, labels[idx, :]), dim=1)
    concat_boxes = torch.cat((boxes, boxes[idx, :]), dim=1)

    return mixed_imgs, concat_labels, concat_boxes


def build_optimizer(cfg, model, method="sgd", no_decay=("bias")):
    decay_params = (p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay))
    no_decay_params = (p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay))
    param_groups = (
        {
            'params': filter(lambda p: p.requires_grad, decay_params),
        },
        {
            'params': filter(lambda p: p.requires_grad, no_decay_params),
            'weight_decay': 0.0,
        },
    )

    if method == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=cfg.init_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    elif method == "adam":
        optimizer = optim.Adam(
            param_groups,
            lr=cfg.init_lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError("unknown optimizer method: {}".format(method))

    return optimizer


def build_scheduler(cfg, optimizer, method="mstep", num_steps=1):
    decay_steps = (cfg.num_epochs - cfg.warmup_epochs) * num_steps - 1

    if method == "mstep":
        scheduler = MultiStepLR(
            optimizer,
            milestones=list(
                map(lambda x: (x - cfg.warmup_epochs) * num_steps,
                    cfg.milestones)),
            gamma=cfg.decay_factor,
        )
    elif method == "exp":
        scheduler = ExponentialLR(
            optimizer,
            gamma=cfg.decay_rate**(1 / (decay_steps)),
        )
    elif method == "cos":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=cfg.final_lr,
        )
    else:
        raise ValueError("unknown scheduler method: {}".format(method))

    return scheduler


class Logger(object):
    def __init__(self, log_path):
        log_name = os.path.basename(log_path)
        self.log_name = log_name if log_name else "root"
        self.log_path = log_path

        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # 配置文件 Handler
        file_handler = logging.FileHandler(self.log_path, "w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

        # 配置屏幕 Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(log_formatter)

        # 添加 handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(cfg):
    """
    在log_dir下, 以当前时间命名, 创建日志文件夹, 并创建logger用于记录训练信息
    :param out_dir: str
    :return:
    """
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, "%Y-%m-%d_%H-%M")
    folder_name = "{}_{}e_{}".format(cfg.data_folder, cfg.num_epochs, time_str)
    log_dir = os.path.join(cfg.ckpt_root_dir, folder_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建logger
    log_path = os.path.join(log_dir, "train.log")
    logger = Logger(log_path)
    logger = logger.init_logger()

    return logger, log_dir


def plot_curve(plt_x, plt_y, mode="loss", name="total", out_dir=None):
    """
    绘制训练和验证集的loss/acc曲线
    :param plt_x: epoch
    :param plt_y: 标量值
    :param mode: 'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(plt_x, plt_y["train"], label="Train")
    plt.plot(plt_x, plt_y["valid"], label="Valid")

    plt.xlabel("epoch")
    plt.ylabel(mode)

    location = "upper right" if mode == "loss" else "upper left"
    plt.legend(loc=location)

    plt.title(" ".join((name, mode)).title())
    plt.savefig(os.path.join(out_dir, "_".join((name, mode)) + ".png"))
    plt.close()


class WarmupLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 warmup_factor=0.001,
                 warmup_iters=1000,
                 warmup_method="linear",
                 last_epoch=-1,
                 verbose=False):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        warmup_factor = self._get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters,
            self.warmup_factor)
        return [base_lr * warmup_factor for base_lr in self.base_lrs]

    def _get_warmup_factor_at_iter(self, method, iter, warmup_iters,
                                   warmup_factor):
        """
        Return the learning rate warmup factor at a specific iteration.
        See https://arxiv.org/abs/1706.02677 for more details.

        Args:
            method (str): warmup method: "constant", "linear" or "cos".
            iter (int): iteration at which to calculate the warmup factor.
            warmup_iters (int): the number of warmup iterations.
            warmup_factor (float): the base warmup factor (the meaning changes according to the method used).

        Returns:
            float: the effective warmup factor at the given iteration.
        """
        if iter >= warmup_iters:
            return 1.0

        if method == "constant":
            return warmup_factor
        elif method == "linear":
            alpha = iter / warmup_iters
            return warmup_factor + (1.0 - warmup_factor) * alpha
        elif method == "cos":
            alpha = iter / warmup_iters
            return warmup_factor + (1.0 - warmup_factor) * (
                1.0 + math.cos(math.pi * (1.0 - alpha))) / 2.0
        else:
            raise ValueError("unknown warmup method: {}".format(method))
