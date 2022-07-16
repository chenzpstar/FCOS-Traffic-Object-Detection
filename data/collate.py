# -*- coding: utf-8 -*-
"""
# @file name  : collate.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-19
# @brief      : 数据打包
"""

import numpy as np
import torch
import torch.nn.functional as F


class Collate():
    def __call__(self, data):
        imgs, labels, boxes = zip(*data)
        assert len(imgs) == len(labels) == len(boxes)

        pad_imgs = []
        pad_labels = []
        pad_boxes = []

        num_h = tuple(map(lambda img: int(img.shape[1]), imgs))
        num_w = tuple(map(lambda img: int(img.shape[2]), imgs))
        num_boxes = tuple(map(lambda boxes: int(boxes.shape[0]), boxes))

        max_h = np.array(num_h).max()
        max_w = np.array(num_w).max()
        max_num_boxes = np.array(num_boxes).max()

        for img, label, box in zip(imgs, labels, boxes):
            pad_imgs.append(
                F.pad(img, (0, int(max_w - img.shape[2]), 0,
                            int(max_h - img.shape[1])),
                      value=0.0))

            pad_labels.append(
                F.pad(label, (0, max_num_boxes - label.shape[0]), value=-1))

            if box.shape[0] != 0:
                pad_boxes.append(
                    F.pad(box, (0, 0, 0, max_num_boxes - box.shape[0]),
                          value=-1))
            else:
                box.unsqueeze_(dim=0)
                pad_boxes.append(
                    F.pad(box, (0, 4, 0, max_num_boxes - box.shape[0]),
                          value=-1))

        return (
            torch.stack(pad_imgs, dim=0),
            torch.stack(pad_labels, dim=0),
            torch.stack(pad_boxes, dim=0),
        )


if __name__ == "__main__":

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, ".."))

    from torch.utils.data import DataLoader

    from bdd100k import BDD100KDataset
    from kitti import KITTIDataset
    from transform import BaseTransform

    size = [800, 1333]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_dir = os.path.join(BASE_DIR, "..", "data", "samples", "kitti")
    train_set = KITTIDataset(data_dir,
                             "training",
                             transform=BaseTransform(size, mean, std))
    train_loader = DataLoader(train_set, batch_size=4, collate_fn=Collate())

    imgs, labels, boxes = next(iter(train_loader))

    print(labels), print(boxes)
    print(imgs.shape, boxes.shape, labels.shape)
