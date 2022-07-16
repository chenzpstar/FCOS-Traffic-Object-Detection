# -*- coding: utf-8 -*-
"""
# @file name  : kitti.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-02
# @brief      : KITTI数据集读取类
"""

import os

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


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
    # classes_list = [
    #     "background", "car", "van", "truck", "pedestrian", "person_sitting",
    #     "cyclist", "tram", "misc"
    # ]
    classes_list = ["background", "car", "pedestrian", "cyclist"]
    num_classes = len(classes_list)

    classes_dict = {name: idx
                    for idx, name in enumerate(classes_list)}  # {name: idx}
    labels_dict = {idx: name
                   for idx, name in enumerate(classes_list)}  # {idx: name}

    def __init__(self,
                 root_dir,
                 set_name,
                 mode="train",
                 split=False,
                 transform=None):
        super(KITTIDataset, self).__init__()
        self.root_dir = root_dir
        self.set_name = set_name
        self.mode = mode
        self.split = split
        self.transform = transform
        self.data_info = []
        self.train_data_info = []
        self.valid_data_info = []
        self._get_data_info()

        if self.split:
            if mode == "train":
                self.data_info = self.train_data_info
            elif mode == "valid":
                self.data_info = self.valid_data_info

        print("loading kitti dataset successfully")

    def __getitem__(self, index):
        # 1. 数据读取
        img_path, anno_path = self.data_info[index]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # bgr -> rgb

        # [idx,], [[x1, y1, x2, y2],]
        labels, boxes = self._get_txt_anno(anno_path)

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
        img_info, anno_info = [], []
        img_dir = os.path.join(self.root_dir, self.set_name, "image_2")

        for img in os.listdir(img_dir):
            if img.endswith(".png"):
                img_path = os.path.join(img_dir, img)
                anno_path = img_path.replace("image_2", "label_2").replace(
                    ".png", ".txt")
                if os.path.isfile(anno_path):
                    img_info.append(img_path)
                    anno_info.append(anno_path)

        if self.split:
            train_img_path, valid_img_path, train_anno_path, valid_anno_path = train_test_split(
                img_info, anno_info, test_size=0.2, random_state=0)
            self.train_data_info = [*zip(train_img_path, train_anno_path)]
            self.valid_data_info = [*zip(valid_img_path, valid_anno_path)]
        else:
            self.data_info = [*zip(img_info, anno_info)]

    def _get_txt_anno(self, txt_path):
        labels, boxes = [], []

        with open(txt_path, 'r') as f:
            for line in f.readlines():
                obj = line.rstrip().split(' ')
                name = obj[0].lower()
                if name in ["car", "van", "truck", "tram"]:
                    labels.append(self.classes_dict["car"])
                elif name in ["pedestrian", "person_sitting"]:
                    labels.append(self.classes_dict["pedestrian"])
                elif name == "cyclist":
                    labels.append(self.classes_dict[name])
                else:
                    continue
                boxes.append(list(map(float, obj[4:8])))

        return np.array(labels), np.array(boxes)


if __name__ == "__main__":

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, ".."))

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    cmap = plt.get_cmap("rainbow")
    colors = list(map(cmap, np.linspace(0, 1, 10)))

    data_dir = os.path.join(BASE_DIR, "..", "data", "samples", "kitti")
    train_set = KITTIDataset(data_dir, "training", mode="train", split=True)
    train_loader = DataLoader(train_set)

    for (img, labels, boxes) in train_loader:
        img = img[0].data.numpy().astype(np.uint8)  # bchw -> chw
        img = img.transpose((1, 2, 0))  # chw -> hwc
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # rgb -> bgr

        labels = labels[0].data.numpy().astype(np.int64)
        boxes = boxes[0].data.numpy().astype(np.int64)

        for label, box in zip(labels, boxes):
            color = tuple(map(lambda i: i * 255, colors[label - 1]))
            cls_name = train_set.labels_dict[label]
            cv2.rectangle(img, box[:2], box[2:], color, 1)
            cv2.rectangle(img, box[:2],
                          (box[0] + len(cls_name) * 10 + 2, box[1] - 20),
                          color, -1)
            cv2.putText(img, cls_name, (box[0] + 2, box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("out", img)
        cv2.waitKey()
