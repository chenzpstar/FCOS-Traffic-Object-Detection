# -*- coding: utf-8 -*-
"""
# @file name  : demo.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-18
# @brief      : FCOS演示
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
from data import BDD100KDataset, KITTIDataset, Normalize, Resize, VOCDataset
from models import FCOSDetector


def get_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--data_folder",
                        default="kitti",
                        type=str,
                        help="dataset folder name")
    parser.add_argument("--ckpt_folder",
                        default="kitti_12e_2022-07-01_14-24",
                        type=str,
                        help="checkpoint folder name")

    return parser.parse_args()


if __name__ == "__main__":
    # 0. config
    args = get_args()

    if args.data_folder == "voc":
        from configs.voc_config import cfg
    elif args.data_folder == "kitti":
        from configs.kitti_config import cfg
    elif args.data_folder == "bdd100k":
        from configs.bdd100k_config import cfg

    cfg.ckpt_folder = args.ckpt_folder if args.ckpt_folder else cfg.ckpt_folder

    cmap = plt.get_cmap("rainbow")
    colors = tuple(map(cmap, np.linspace(0, 1, 20)))

    data_dir = os.path.join(BASE_DIR, "..", "..", "datasets", cfg.data_folder)
    assert os.path.exists(data_dir)

    ckpt_dir = os.path.join(BASE_DIR, "..", "..", "results", cfg.ckpt_folder)
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
    assert os.path.exists(ckpt_path)

    # 1. data
    if cfg.data_folder == "voc":
        img_dir = os.path.join(data_dir, "VOC2007", "JPEGImages")
        dataset = VOCDataset
    elif cfg.data_folder == "kitti":
        img_dir = os.path.join(data_dir, "testing", "image_2")
        dataset = KITTIDataset
    elif cfg.data_folder == "bdd100k":
        img_dir = os.path.join(data_dir, "images", "100k", "test")
        dataset = BDD100KDataset

    # 2. model
    model = FCOSDetector(cfg).to(cfg.device)
    model_weights = torch.load(ckpt_path, map_location=cfg.device)
    model_dict = dict(zip(model.state_dict().keys(), model_weights.values()))
    model.load_state_dict(model_dict)
    model.eval()
    print("loading model successfully")

    # 3. demo
    for img in os.listdir(img_dir):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        img_path = os.path.join(img_dir, img)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_pad = Resize(cfg.size)(img_bgr)
        img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        img_norm = Normalize(cfg.mean, cfg.std)(img_rgb)
        img_chw = img_norm.transpose((2, 0, 1))  # hwc -> chw
        img_tensor = torch.from_numpy(img_chw).float()
        img_tensor.unsqueeze_(dim=0)  # chw -> bchw

        with torch.no_grad():
            img_tensor = img_tensor.to(cfg.device, non_blocking=True)
            scores, labels, boxes = model(img_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cost_time = time.time() - start_time
        print("T: {:.3f} ms, FPS: {:.3f}".format(cost_time * 1000,
                                                 1.0 / cost_time))

        scores = scores[0].cpu().numpy().astype(np.float32)
        labels = labels[0].cpu().numpy().astype(np.int64)
        boxes = boxes[0].cpu().numpy().astype(np.int64)

        for score, label, box in zip(scores, labels, boxes):
            color = tuple(map(lambda i: i * 255, colors[label - 1]))
            cls_name = dataset.classes_list[label]
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
