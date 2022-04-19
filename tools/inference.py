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
# from configs.bdd100k_config import cfg
from configs.kitti_config import cfg
from data import BDD100KDataset, KITTIDataset, Normalize, Resize
from models.fcos import FCOSDetector

# 添加解析参数
parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--data_folder",
                    default="kitti",
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

    size = [800, 1333]
    mean = [0.3665, 0.3857, 0.3744]  # [0.485, 0.456, 0.406]
    std = [0.3160, 0.3205, 0.3262]  # [0.229, 0.224, 0.225]

    data_dir = os.path.join(BASE_DIR, "..", "..", "datasets", "kitti")
    assert os.path.exists(data_dir)

    # 1. dataset
    img_dir = os.path.join(data_dir, "testing", "image_2")

    # 2. model
    model = FCOSDetector(mode="inference", cfg=cfg)
    ckpt_folder = "kitti_12e_2022-04-19_18-42"
    ckpt_path = os.path.join(BASE_DIR, "..", "..", "results", ckpt_folder,
                             "checkpoint_12.pth")
    model_weights = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model_weights = {
        k: model_weights[k] if k in model_weights else model.state_dict()[k]
        for k in model.state_dict()
    }
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()
    print("INFO ==> finish loading model")

    # 3. inference
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_pad = Resize(size)(img_bgr)
        img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        img_norm = Normalize(mean, std)(img_rgb)
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

        scores = scores[0].cpu().numpy().astype(np.float64)
        labels = labels[0].cpu().numpy().astype(np.int64)
        boxes = boxes[0].cpu().numpy().astype(np.int64)

        for score, label, box in zip(scores, labels, boxes):
            color = [i * 255 for i in colors[label - 1]]
            cls_name = KITTIDataset.labels_dict[label]
            cv2.rectangle(img_pad, box[:2], box[2:], color, 1)
            cv2.rectangle(img_pad, box[:2],
                          (box[0] + len(cls_name) * 10 + 72, box[1] - 20),
                          color, -1)
            cv2.putText(img_pad, "{}: {:.4f}".format(cls_name, score),
                        (box[0] + 2, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            print("{}: {:.4f}".format(cls_name, score))

        cv2.imshow("out", img_pad)
        cv2.waitKey()
