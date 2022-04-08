# -*- coding: utf-8 -*-
# common.py

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
        self.out_path = log_path

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, "w")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(out_dir, cfg):
    """
    在out_dir文件夹下以当前时间命名，创建日志文件夹，并创建logger用于记录信息
    :param out_dir: str
    :return:
    """
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, "%Y-%m-%d_%H-%M")
    folder_name = "{}_{}e_{}".format(cfg.data_folder, cfg.max_epoch, time_str)
    log_dir = os.path.join(out_dir, folder_name)  # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建logger
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()

    return logger, log_dir


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label="Train")
    plt.plot(valid_x, valid_y, label="Valid")

    plt.ylabel(str(mode))
    plt.xlabel("Epoch")

    location = "upper right" if mode == "loss" else "upper left"
    plt.legend(loc=location)

    plt.title("_".join([mode]))
    plt.savefig(os.path.join(out_dir, mode + ".png"))
    plt.close()
