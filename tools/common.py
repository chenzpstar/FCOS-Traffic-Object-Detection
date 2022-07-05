# -*- coding: utf-8 -*-
"""
# @file name  : common.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-16
# @brief      : 通用函数
"""

import logging
import os
import random
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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

        # 配置文件Handler
        file_handler = logging.FileHandler(self.log_path, "w")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_formatter = logging.Formatter(
        #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # console_handler.setFormatter(console_formatter)

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(cfg):
    """
    在log_dir文件夹下以当前时间命名, 创建日志文件夹, 并创建logger用于记录信息
    :param out_dir: str
    :return:
    """
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, "%Y-%m-%d_%H-%M")
    folder_name = "{}_{}e_{}".format(cfg.data_folder, cfg.max_epoch, time_str)
    log_dir = os.path.join(cfg.ckpt_root_dir, folder_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建logger
    log_path = os.path.join(log_dir, "train.log")
    logger = Logger(log_path)
    logger = logger.init_logger()

    return logger, log_dir


def plot_curve(train_x,
               train_y,
               valid_x,
               valid_y,
               mode="loss",
               kind="total",
               out_dir=None):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode: 'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label="Train")
    plt.plot(valid_x, valid_y, label="Valid")

    plt.xlabel("epoch")
    plt.ylabel(mode)

    location = "upper right" if mode == "loss" else "upper left"
    plt.legend(loc=location)

    plt.title(" ".join([kind, mode]).title())
    plt.savefig(os.path.join(out_dir, "_".join([kind, mode]) + ".png"))
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
            method (str): warmup method; either "constant" or "linear".
            iter (int): iteration at which to calculate the warmup factor.
            warmup_iters (int): the number of warmup iterations.
            warmup_factor (float): the base warmup factor (the meaning changes according
                to the method used).

        Returns:
            float: the effective warmup factor at the given iteration.
        """
        if iter >= warmup_iters:
            return 1.0

        if method == "constant":
            return warmup_factor
        elif method == "linear":
            alpha = iter / warmup_iters
            return warmup_factor * (1.0 - alpha) + alpha
        else:
            raise ValueError("unknown warmup method: {}".format(method))
