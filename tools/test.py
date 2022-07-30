# -*- coding: utf-8 -*-
"""
# @file name  : test.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-17
# @brief      : FCOS测试
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import numpy as np
import torch
# from configs.bdd100k_config import cfg
from configs.kitti_config import cfg
from data import BDD100KDataset, Collate, KITTIDataset
from models import FCOSDetector
from torch.utils.data import DataLoader

from eval import eval_model

# 添加解析参数
parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--bs", default=None, type=int, help="batch size")
parser.add_argument("--data_folder",
                    default="kitti",
                    type=str,
                    help="dataset folder name")
parser.add_argument("--ckpt_folder",
                    default="kitti_12e_2022-07-01_14-24",
                    type=str,
                    help="checkpoint folder name")
args = parser.parse_args()

# 修改配置参数
cfg.valid_bs = args.bs if args.bs else cfg.valid_bs
cfg.data_folder = args.data_folder if args.data_folder else cfg.data_folder
cfg.ckpt_folder = args.ckpt_folder if args.ckpt_folder else cfg.ckpt_folder

if __name__ == "__main__":
    # 0. config
    data_dir = os.path.join(BASE_DIR, "..", "..", "datasets", cfg.data_folder)
    assert os.path.exists(data_dir)

    ckpt_dir = os.path.join(BASE_DIR, "..", "..", "results", cfg.ckpt_folder)
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
    assert os.path.exists(ckpt_path)

    out_path = os.path.join(ckpt_dir, "eval.txt")

    # 1. dataset
    if cfg.data_folder == "kitti":
        test_set = KITTIDataset(
            data_dir,
            set_name="training",
            mode="valid",
            split=True,
            transform=cfg.base_tf,
        )
    elif cfg.data_folder == "bdd100k":
        test_set = BDD100KDataset(
            data_dir,
            set_name="val",
            transform=cfg.base_tf,
        )
    print("test set has {} imgs".format(len(test_set)))

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=Collate(),
        pin_memory=True,
    )
    print("test loader has {} iters".format(len(test_loader)))

    # 2. model
    model = FCOSDetector(cfg).to(cfg.device)
    model_weights = torch.load(ckpt_path, map_location=cfg.device)
    model_dict = dict(zip(model.state_dict().keys(), model_weights.values()))
    model.load_state_dict(model_dict)
    model.eval()
    print("loading model successfully")

    # 3. test
    for thr in (0.5, 0.75):
        # 评估指标
        metrics = eval_model(model, test_loader, cfg.num_classes, thr,
                             cfg.use_07_metric, cfg.device)

        # 计算 mAP
        mAP = np.mean(metrics["ap"])

        # 输出结果
        with open(out_path, "a") as f:
            for label in range(cfg.num_classes):
                print(
                    "class: {}, rec: {:.3%}, prec: {:.3%}, f1: {:.3%}, ap: {:.3%}"
                    .format(test_set.classes_list[label + 1],
                            metrics["rec"][label], metrics["prec"][label],
                            metrics["f1"][label], metrics["ap"][label]),
                    file=f)
            print("mAP@{}: {:.3%}".format(thr, mAP), file=f)
