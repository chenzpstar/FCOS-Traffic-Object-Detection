# -*- coding: utf-8 -*-
"""
# @file name  : voc.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-09-30
# @brief      : VOC数据集读取类
"""

import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    """
    - voc
        - VOC2007
            - JPEGImages
                - 000001.jpg
            - Annotations
                - 000001.xml
            - ImageSets
                - Main
                    - trainval
                    - train
                    - val
                    - test
        - VOC2012
            - JPEGImages
            - Annotations
            - ImageSets
    """
    classes_list = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
        "tvmonitor"
    ]
    # classes_list = [
    #     "background", "car", "bus", "bicycle", "motorbike", "person", "train"
    # ]
    num_classes = len(classes_list)

    classes_dict = dict(zip(classes_list, range(num_classes)))  # {name: idx}

    def __init__(self,
                 root_dir,
                 year="2007",
                 set_name="trainval",
                 use_difficult=False,
                 transform=None):
        super(VOCDataset, self).__init__()
        self.root_dir = root_dir
        self.folder_name = "VOC" + year
        self.set_file = set_name + ".txt"
        self.use_difficult = use_difficult
        self.transform = transform
        self.data_info = []
        self._get_data_info()
        print("loading voc dataset successfully")

    def __getitem__(self, index):
        # 1. 数据读取
        img_path, anno_path = self.data_info[index]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # bgr -> rgb

        # [idx,], [[x1, y1, x2, y2],]
        labels, boxes = self._get_xml_anno(anno_path)

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
        img_dir = os.path.join(self.root_dir, self.folder_name, "JPEGImages")
        set_path = os.path.join(self.root_dir, self.folder_name, "ImageSets",
                                "Main", self.set_file)

        with open(set_path, 'r') as f:
            for line in f.readlines():
                img_path = os.path.join(img_dir, line.strip() + ".jpg")
                anno_path = img_path.replace("JPEGImages",
                                             "Annotations").replace(
                                                 ".jpg", ".xml")
                if os.path.isfile(anno_path):
                    self.data_info.append((img_path, anno_path))

        random.shuffle(self.data_info)

    def _get_xml_anno(self, xml_path):
        labels, boxes = [], []
        anno = ET.parse(xml_path).getroot()

        for obj in anno.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.use_difficult and difficult:
                continue

            name = obj.find("name").text.lower().strip()
            labels.append(self.classes_dict[name])

            _box = obj.find("bndbox")
            box = (
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            )
            TO_REMOVE = 1
            box = tuple(map(lambda x: x - TO_REMOVE, tuple(map(float, box))))
            boxes.append(box)

        return np.array(labels), np.array(boxes)


if __name__ == "__main__":

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, ".."))

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    cmap = plt.get_cmap("rainbow")
    colors = tuple(map(cmap, np.linspace(0, 1, 20)))

    data_dir = os.path.join(BASE_DIR, "samples", "voc")
    train_set = VOCDataset(data_dir, "2007", "train")
    train_loader = DataLoader(train_set)

    for (img, labels, boxes) in train_loader:
        img = img[0].data.numpy().astype(np.uint8)  # bchw -> chw
        img = img.transpose((1, 2, 0))  # chw -> hwc
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # rgb -> bgr

        labels = labels[0].data.numpy().astype(np.int64)
        boxes = boxes[0].data.numpy().astype(np.int64)

        for label, box in zip(labels, boxes):
            color = tuple(map(lambda i: i * 255, colors[label - 1]))
            cls_name = train_set.classes_list[label]
            cv2.rectangle(img, box[:2], box[2:], color, 1)
            cv2.rectangle(img, box[:2],
                          (box[0] + len(cls_name) * 10 + 2, box[1] - 20),
                          color, -1)
            cv2.putText(img, cls_name, (box[0] + 2, box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("out", img)
        cv2.waitKey()
