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
from data.bdd100k import BDD100KDataset
from data.collate import Collate
from data.kitti import KITTIDataset
from data.transform import BaseTransform
from models.fcos import FCOSDetector
from torch.utils.data import DataLoader

from eval import evalate_ap

# 添加解析参数
parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--data_folder",
                    default="bdd100k",
                    type=str,
                    help="dataset folder name")
parser.add_argument("--ckpt_folder",
                    default=None,
                    type=str,
                    help="checkpoint folder name")
args = parser.parse_args()

if __name__ == "__main__":
    # 0. config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    size = [800, 1333]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_dir = os.path.join(BASE_DIR, "..", "..", "datasets", "bdd100k")
    assert os.path.exists(data_dir)

    # 1. dataset
    test_set = BDD100KDataset(data_dir,
                              set_name="test",
                              transform=BaseTransform(size, mean, std))
    print("INFO ==> test dataset has {} imgs".format(len(test_set)))

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=Collate(),
    )

    # 2. model
    model = FCOSDetector(mode="inference")
    checkpoint = torch.load("./results/checkpoint_24.pth",
                            map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("INFO ==> finish loading model")

    # 3. test
    # 计算AP
    all_ap = evalate_ap(test_set, test_loader, model, device=device)
    print("all classes AP:")
    for label, ap in all_ap.items():
        print('ap for {} is {}'.format(test_set.labels_dict[int(label)], ap))

    # 计算mAP
    mAP = sum(all_ap.values()) / (test_set.cls_num - 1)
    print("mAP: {:.4f}".format(mAP))
