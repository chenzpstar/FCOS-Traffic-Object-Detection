# -*- coding: utf-8 -*-
"""
# @file name  : kitti.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-02
# @brief      : KITTI数据集读取类
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class KITTIDataset(Dataset):
    """
    KITTI dataset

    - kitti
        - training
            - image_2
                - 000000.png
            - label_2
                - 000000.txt
        - testing
            - image_2
    """
    cls_names = [
        "Background", "Car", "Van", "Truck", "Pedestrian", "Person_sitting",
        "Cyclist", "Tram", "Misc"
    ]
    cls_num = len(cls_names)

    cls_names_dict = {name: idx
                      for idx, name in enumerate(cls_names)}  # {name: idx}
    labels_dict = {idx: name
                   for idx, name in enumerate(cls_names)}  # {idx: name}

    def __init__(self, root_dir, set_name, transform=None):
        super(KITTIDataset, self).__init__()
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.data_info = []
        self._get_data_info()
        print("INFO ==> finish loading kitti dataset")

    def __getitem__(self, index):
        # step 1: 数据读取
        img_path, anno_path = self.data_info[index]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # bgr -> rgb

        # [idx,], [[x1, y1, x2, y2],]
        labels, boxes = self._get_txt_anno(anno_path)

        # step 2: 数据预处理
        if self.transform is not None:
            img_rgb, boxes = self.transform(img_rgb, boxes)

        # step 3: 数据格式转换
        img_chw = img_rgb.transpose((2, 0, 1))  # hwc -> chw
        img_tensor = torch.from_numpy(img_chw).float()

        labels_tensor = torch.from_numpy(labels).long()
        boxes_tensor = torch.from_numpy(boxes).float()

        return img_tensor, labels_tensor, boxes_tensor

    def __len__(self):
        assert len(
            self.data_info) > 0, "INFO ==> please check your path to dataset"
        return len(self.data_info)

    def _get_data_info(self):
        img_dir = os.path.join(self.root_dir, self.set_name, "image_2")
        for img in os.listdir(img_dir):
            if img.endswith(".png"):
                img_path = os.path.join(img_dir, img)
                anno_path = img_path.replace("image_2", "label_2").replace(
                    ".png", ".txt")
                if os.path.isfile(anno_path):
                    self.data_info.append((img_path, anno_path))
        random.shuffle(self.data_info)

    def _get_txt_anno(self, txt_path):
        labels, boxes = [], []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                obj = line.rstrip().split(' ')
                if obj[0] in self.cls_names:
                    labels.append(self.cls_names_dict[obj[0]])
                    boxes.append(list(map(float, obj[4:8])))

        return np.array(labels), np.array(boxes)


if __name__ == "__main__":

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, ".."))

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    data_dir = os.path.join(BASE_DIR, "..", "data", "samples", "kitti")
    train_set = KITTIDataset(data_dir, "training")
    train_loader = DataLoader(train_set)

    img, labels, boxes = next(iter(train_loader))

    img = img.squeeze(0).data.numpy().astype(np.uint8)  # bchw -> chw
    img = img.transpose((1, 2, 0))  # chw -> hwc
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # rgb -> bgr

    labels = labels.squeeze(0).data.numpy().astype(np.int64)
    boxes = boxes.squeeze(0).data.numpy().astype(np.int64)

    for label, box in zip(labels, boxes):
        color = [i * 255 for i in colors[int(label) - 1]]
        cls_name = train_set.labels_dict[int(label)]
        cv2.rectangle(img, box[:2], box[2:], color, 2)
        cv2.rectangle(img, box[:2], (box[0] + len(cls_name) * 15, box[1] - 25),
                      color, -1)
        cv2.putText(img, cls_name, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    cv2.imshow("out", img)
    cv2.waitKey()
