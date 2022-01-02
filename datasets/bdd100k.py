# -*- coding: utf-8 -*-
"""
# @file name  : bdd100k.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-01
# @brief      : bdd100k数据集读取类
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
                    - 0000f77c-62c2a288.jpg
                - val
                - test
        - labels
            - 100k
                - train
                    - 0000f77c-62c2a288.json
                - val
    """
    cls_names = [
        "car", "bus", "truck", "motor", "bike", "person", "rider", "train"
    ]
    cls_num = len(cls_names)

    cls_names_dict = dict(zip(cls_names, range(1, cls_num + 1)))  # {name: id}
    cls_ids_dict = dict(zip(range(1, cls_num + 1), cls_names))  # {id: name}

    def __init__(self, root_dir, set_name, transform=None):
        super(BDD100KDataset, self).__init__()
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.data_info = []
        self._get_data_info()
        # self._collate_fn(self.data_info)

    def __getitem__(self, index):
        # step 1: 数据读取
        img_path, label_path = self.data_info[index]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # bgr -> rgb

        # [[id, x1, y1, x2, y2],]
        labels = self._get_json_label(label_path)
        cls_ids = labels[..., 0]
        boxes = labels[..., 1:]

        # step 2: 数据预处理
        if self.transform is not None:
            img_rgb, boxes = self.transform(img_rgb, boxes)

        # step 3: 数据格式转换
        img_chw = img_rgb.transpose((2, 0, 1))  # hwc -> chw
        img_tensor = torch.from_numpy(img_chw).float()

        cls_ids_tensor = torch.tensor(cls_ids, dtype=torch.float)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float)

        return img_tensor, cls_ids_tensor, boxes_tensor

    def __len__(self):
        assert len(self.data_info) > 0, "Please check your path to dataset!"
        return len(self.data_info)

    def _get_data_info(self):
        img_dir = os.path.join(self.root_dir, "images", "100k", self.set_name)
        img_path = [
            os.path.join(img_dir, img) for img in os.listdir(img_dir)
            if img.endswith(".jpg")
        ]
        self.data_info = [
            (path, path.replace("images", "labels").replace(".jpg", ".json"))
            for path in img_path
        ]
        random.shuffle(self.data_info)

    def _get_json_label(self, json_path):
        with open(json_path, 'r') as f:
            labels = []
            anno = json.load(f)
            objs = anno["frames"][0]["objects"]
            for obj in objs:
                if obj["category"] in self.cls_names:
                    labels.append([
                        self.cls_names_dict[obj["category"]],
                        obj["box2d"]["x1"],
                        obj["box2d"]["y1"],
                        obj["box2d"]["x2"],
                        obj["box2d"]["y2"],
                    ])

        return np.asarray(labels, dtype=float)

    # def _collate_fn(self, data):
    #     pass


if __name__ == "__main__":

    data_dir = os.path.join("examples", "bdd100k")
    train_set = BDD100KDataset(data_dir, "train")
    train_loader = DataLoader(train_set)

    img, cls_ids, boxes = next(iter(train_loader))

    img = img.squeeze(0).data.numpy().astype(np.uint8)  # bchw -> chw
    img = img.transpose((1, 2, 0))  # chw -> hwc
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # rgb -> bgr

    cls_ids = cls_ids.squeeze(0).data.numpy().astype(int)
    boxes = boxes.squeeze(0).data.numpy().astype(int)

    for cls_id, box in zip(cls_ids, boxes):
        cls_name = train_set.cls_ids_dict[cls_id]
        cv2.rectangle(img, box[:2], box[2:], (0, 0, 255), 2)
        cv2.rectangle(img, box[:2], (box[0] + len(cls_name) * 15, box[1] - 25),
                      (0, 0, 255), -1)
        cv2.putText(img, cls_name, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    cv2.imshow("out", img)
    cv2.waitKey()
