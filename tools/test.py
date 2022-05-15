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

import torch
# from configs.bdd100k_config import cfg
from configs.kitti_config import cfg
from data import BaseTransform, BDD100KDataset, Collate, KITTIDataset
from models.fcos import FCOSDetector
from torch.utils.data import DataLoader

from eval import eval_model

# 添加解析参数
parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--data_folder",
                    default="kitti",
                    type=str,
                    help="dataset folder name")
parser.add_argument("--ckpt_folder",
                    default="kitti_12e_2022-05-12_22-24",
                    type=str,
                    help="checkpoint folder name")
args = parser.parse_args()

# 修改配置参数
cfg.data_folder = args.data_folder if args.data_folder else cfg.data_folder
cfg.ckpt_folder = args.ckpt_folder if args.ckpt_folder else cfg.ckpt_folder

if __name__ == "__main__":
    # 0. config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    size = [800, 1333]
    mean = [0.3665, 0.3857, 0.3744]  # [0.485, 0.456, 0.406]
    std = [0.3160, 0.3205, 0.3262]  # [0.229, 0.224, 0.225]

    data_dir = os.path.join(BASE_DIR, "..", "..", "datasets", cfg.data_folder)
    assert os.path.exists(data_dir)

    ckpt_dir = os.path.join(BASE_DIR, "..", "..", "results")
    ckpt_path = os.path.join(ckpt_dir, cfg.ckpt_folder, "checkpoint_12.pth")
    assert os.path.exists(ckpt_path)

    # 1. dataset
    test_set = KITTIDataset(
        data_dir,
        set_name="training",
        mode="valid",
        split=True,
        transform=BaseTransform(size, mean, std),
    )
    print("INFO ==> test set has {} imgs".format(len(test_set)))

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=Collate(),
    )

    # 2. model
    model = FCOSDetector(mode="inference", cfg=cfg)
    model_weights = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model_weights = {
        k: model_weights[k] if k in model_weights else model.state_dict()[k]
        for k in model.state_dict()
    }
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()
    print("INFO ==> finish loading model")

    # 3. test
    # 评估指标
    recalls, precisions, f1s, aps = eval_model(
        test_set,
        test_loader,
        model,
        device=device,
    )

    # 计算mAP
    mAP = sum(aps) / (test_set.cls_num - 1)

    # 输出结果
    out_path = os.path.join(ckpt_dir, cfg.ckpt_folder, "eval.txt")
    with open(out_path, "w") as f:
        for label in range(test_set.cls_num - 1):
            print(
                "class: {}, recall: {:.4f}, precision: {:.4f}, f1: {:.4f}, ap: {:.4f}"
                .format(test_set.labels_dict[label + 1], recalls[label],
                        precisions[label], f1s[label], aps[label]),
                file=f)
        print("mAP: {:.4f}".format(mAP), file=f)
