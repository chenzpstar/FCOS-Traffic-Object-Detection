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
from models import FCOSDetector

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
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(BASE_DIR, "..", "..", "datasets", cfg.data_folder)
    assert os.path.exists(data_dir)

    ckpt_dir = os.path.join(BASE_DIR, "..", "..", "results")
    ckpt_path = os.path.join(ckpt_dir, cfg.ckpt_folder, "checkpoint_best.pth")
    assert os.path.exists(ckpt_path)

    # 1. dataset
    if cfg.data_folder == "kitti":
        img_dir = os.path.join(data_dir, "testing", "image_2")
    elif cfg.data_folder == "bdd100k":
        img_dir = os.path.join(data_dir, "images", "100k", "test")

    # 2. model
    model = FCOSDetector(mode="inference", cfg=cfg)
    model_weights = torch.load(ckpt_path, map_location=torch.device("cpu"))
    state_dict = {
        k: v
        for k, v in zip(model.state_dict(), model_weights.values())
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("loading model successfully")

    # 3. inference
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_pad = Resize(cfg.size)(img_bgr)
        img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        img_norm = Normalize(cfg.mean, cfg.std)(img_rgb)
        img_chw = img_norm.transpose((2, 0, 1))  # hwc -> chw
        img_tensor = torch.from_numpy(img_chw).float()
        img_tensor = img_tensor.unsqueeze_(dim=0).to(device)  # chw -> bchw

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            scores, labels, boxes = model(img_tensor)

        torch.cuda.synchronize()
        cost_time = int((time.time() - start_time) * 1000)
        print("processing img done, cost time: {} ms".format(cost_time))

        scores = scores[0].cpu().numpy().astype(np.float32)
        labels = labels[0].cpu().numpy().astype(np.int64)
        boxes = boxes[0].cpu().numpy().astype(np.int64)

        for score, label, box in zip(scores, labels, boxes):
            color = [i * 255 for i in colors[label - 1]]
            cls_name = KITTIDataset.labels_dict[label]
            cv2.rectangle(img_pad, box[:2], box[2:], color, 1)
            cv2.rectangle(img_pad, box[:2],
                          (box[0] + len(cls_name) * 10 + 72, box[1] - 20),
                          color, -1)
            cv2.putText(img_pad, "{}: {:.3%}".format(cls_name, score),
                        (box[0] + 2, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            print("{}: {:.3%}".format(cls_name, score))

        cv2.imshow("out", img_pad)
        cv2.waitKey()
