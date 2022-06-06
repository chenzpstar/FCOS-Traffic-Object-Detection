# -*- coding: utf-8 -*-
"""
# @file name  : bdd100k.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-01
# @brief      : BDD100K数据集读取类
"""

import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BDD100KDataset(Dataset):
    """
    BDD100K dataset

    - bdd100k
        - images
            - 100k
                - train
                    - 0000f77c-6257be58.jpg
                - val
                - test
        - labels
            - 100k
                - train
                    - 0000f77c-6257be58.json
                - val
    """
    # classes_list = [
    #     "background", "car", "bus", "truck", "motor", "bike", "pedestrian",
    #     "rider", "train"
    # ]
    classes_list = ["background", "car", "pedestrian", "rider"]
    num_classes = len(classes_list)

    classes_dict = {name: idx
                    for idx, name in enumerate(classes_list)}  # {name: idx}
    labels_dict = {idx: name
                   for idx, name in enumerate(classes_list)}  # {idx: name}

    def __init__(self, root_dir, set_name, transform=None):
        super(BDD100KDataset, self).__init__()
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.data_info = []
        self._get_data_info()
        print("loading bdd100k dataset successfully")

    def __getitem__(self, index):
        # 1. 数据读取
        img_path, anno_path = self.data_info[index]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # bgr -> rgb

        # [idx,], [[x1, y1, x2, y2],]
        labels, boxes = self._get_json_anno(anno_path)

        # 2. 数据预处理
        if self.transform is not None:
            img_rgb, boxes = self.transform(img_rgb, boxes)

        # 3. 数据格式转换
        img_chw = img_rgb.transpose((2, 0, 1))  # hwc -> chw
        img_tensor = torch.from_numpy(img_chw).float()

        labels_tensor = torch.from_numpy(labels).long()
        boxes_tensor = torch.from_numpy(boxes).float()

        return img_tensor, labels_tensor, boxes_tensor

    def __len__(self):
        assert len(self.data_info) > 0, "please check your path to dataset"
        return len(self.data_info)

    def _get_data_info(self):
        img_dir = os.path.join(self.root_dir, "images", "100k", self.set_name)
        for img in os.listdir(img_dir):
            if img.endswith(".jpg"):
                img_path = os.path.join(img_dir, img)
                anno_path = img_path.replace("images", "labels").replace(
                    ".jpg", ".json")
                if os.path.isfile(anno_path):
                    self.data_info.append((img_path, anno_path))
        random.shuffle(self.data_info)

    def _get_json_anno(self, json_path):
        labels, boxes = [], []
        with open(json_path, 'r') as f:
            anno = json.load(f)
            try:
                objs = anno["labels"]
            except KeyError:
                pass
            else:
                for obj in objs:
                    name = obj["category"]
                    if name in ["car", "bus", "truck", "motor", "train"]:
                        labels.append(self.classes_dict["car"])
                    elif name in ["pedestrian", "rider"]:
                        labels.append(self.classes_dict[name])
                    else:
                        continue
                    boxes.append([
                        obj["box2d"]["x1"],
                        obj["box2d"]["y1"],
                        obj["box2d"]["x2"],
                        obj["box2d"]["y2"],
                    ])

        return np.array(labels), np.array(boxes)


if __name__ == "__main__":

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, ".."))

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    data_dir = os.path.join(BASE_DIR, "..", "data", "samples", "bdd100k")
    train_set = BDD100KDataset(data_dir, "train")
    train_loader = DataLoader(train_set)

    for (img, labels, boxes) in train_loader:
        img = img.squeeze(0).data.numpy().astype(np.uint8)  # bchw -> chw
        img = img.transpose((1, 2, 0))  # chw -> hwc
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # rgb -> bgr

        labels = labels.squeeze(0).data.numpy().astype(np.int64)
        boxes = boxes.squeeze(0).data.numpy().astype(np.int64)

        for label, box in zip(labels, boxes):
            color = [i * 255 for i in colors[label - 1]]
            cls_name = train_set.labels_dict[label]
            cv2.rectangle(img, box[:2], box[2:], color, 1)
            cv2.rectangle(img, box[:2],
                          (box[0] + len(cls_name) * 10 + 2, box[1] - 20),
                          color, -1)
            cv2.putText(img, cls_name, (box[0] + 2, box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("out", img)
        cv2.waitKey()
