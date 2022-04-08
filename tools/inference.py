# -*- coding: utf-8 -*-
"""
# @file name  : inference.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-18
# @brief      : FCOS推理
"""

import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from data.bdd100k import BDD100KDataset
from data.kitti import KITTIDataset
from data.transform import Normalize, Resize
from models.fcos import FCOSDetector

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
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(BASE_DIR, "..", "..", "datasets", "bdd100k")
    assert os.path.exists(data_dir)

    # 1. dataset
    img_dir = os.path.join(data_dir, "images", "100k", "test")

    # 2. model
    model = FCOSDetector(mode="inference")
    checkpoint = torch.load("./results/checkpoint_24.pth",
                            map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("INFO ==> finish loading model")

    # 3. inference
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_pad = Resize()(img_rgb)
        img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        img_norm = Normalize()(img_rgb)
        img_chw = img_norm.transpose((2, 0, 1))  # hwc -> chw
        img_tensor = torch.from_numpy(img_chw).float()
        img_tensor = img_tensor.unsqueeze(dim=0).to(device)  # chw -> bchw

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            scores, labels, boxes = model(img_tensor)

        torch.cuda.synchronize()
        cost_time = int((time.time() - start_time) * 1000)
        print("INFO ==> finish processing img, cost time: {} ms".format(
            cost_time))

        scores = scores[0].cpu().numpy().tolist()
        labels = labels[0].cpu().numpy().tolist()
        boxes = boxes[0].cpu().numpy().tolist()

        for score, label, box in zip(scores, labels, boxes):
            color = [i * 255 for i in colors[int(label) - 1]]
            cls_name = BDD100KDataset.labels_dict[int(label)]
            cv2.rectangle(img_pad, box[:2], box[2:], color, 2)
            cv2.rectangle(img_pad, box[:2],
                          (box[0] + len(cls_name) * 15, box[1] - 25), color,
                          -1)
            cv2.putText(img_pad, "{}:{:.4f}".format(cls_name, score),
                        (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 0), 2)
            print(cls_name, score)

        cv2.imshow("out", img_pad)
        cv2.waitKey()
