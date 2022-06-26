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
parser.add_argument("--data_folder",
                    default="kitti",
                    type=str,
                    help="dataset folder name")
parser.add_argument("--ckpt_folder",
                    default="kitti_12e_2022-06-01_16-01",
                    type=str,
                    help="checkpoint folder name")
args = parser.parse_args()

# 修改配置参数
cfg.data_folder = args.data_folder if args.data_folder else cfg.data_folder
cfg.ckpt_folder = args.ckpt_folder if args.ckpt_folder else cfg.ckpt_folder

if __name__ == "__main__":
    # 0. config
    data_dir = os.path.join(BASE_DIR, "..", "..", "datasets", cfg.data_folder)
    assert os.path.exists(data_dir)

    ckpt_dir = os.path.join(BASE_DIR, "..", "..", "results")
    ckpt_path = os.path.join(ckpt_dir, cfg.ckpt_folder, "checkpoint_best.pth")
    assert os.path.exists(ckpt_path)

    out_path = os.path.join(ckpt_dir, cfg.ckpt_folder, "eval.txt")

    # 1. dataset
    if cfg.data_folder == "kitti":
        test_set = KITTIDataset(
            data_dir,
            set_name="training",
            mode="valid",
            split=True,
            transform=cfg.base_trans,
        )
    elif cfg.data_folder == "bdd100k":
        test_set = BDD100KDataset(
            data_dir,
            set_name="val",
            transform=cfg.base_trans,
        )
    print("test set has {} imgs".format(len(test_set)))

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=Collate(),
    )
    print("test loader has {} iters".format(len(test_loader)))

    # 2. model
    model = FCOSDetector(cfg)
    model_weights = torch.load(ckpt_path, map_location=torch.device("cpu"))
    state_dict = {
        k: v
        for k, v in zip(model.state_dict(), model_weights.values())
    }
    model.load_state_dict(state_dict)
    model.to(cfg.device)
    model.eval()
    print("loading model successfully")

    # 3. test
    for thr in [0.5, 0.75]:
        # 评估指标
        recalls, precisions, f1s, aps = eval_model(
            model,
            test_loader,
            num_classes=cfg.num_classes,
            iou_thr=thr,
            device=cfg.device,
        )

        # 计算mAP
        mAP = np.mean(aps)

        # 输出结果
        with open(out_path, "a") as f:
            for label in range(cfg.num_classes):
                print(
                    "class: {}, recall: {:.3%}, precision: {:.3%}, f1: {:.3%}, ap: {:.3%}"
                    .format(test_set.labels_dict[label + 1], recalls[label],
                            precisions[label], f1s[label], aps[label]),
                    file=f)
            print("mAP: {:.3%}".format(mAP), file=f)
