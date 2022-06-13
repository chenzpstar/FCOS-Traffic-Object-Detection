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


class Collate:
    def __call__(self, data):
        imgs_list, labels_list, boxes_list = zip(*data)
        assert len(imgs_list) == len(labels_list) == len(boxes_list)

        pad_imgs = []
        pad_labels = []
        pad_boxes = []

        h_list = [int(img.shape[1]) for img in imgs_list]
        w_list = [int(img.shape[2]) for img in imgs_list]
        boxes_num_list = [int(boxes.shape[0]) for boxes in boxes_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        max_boxes_num = np.array(boxes_num_list).max()

        for img, labels, boxes in zip(imgs_list, labels_list, boxes_list):
            pad_imgs.append(
                F.pad(img, (0, int(max_w - img.shape[2]), 0,
                            int(max_h - img.shape[1])),
                      value=0.0))
            pad_labels.append(
                F.pad(labels, (0, max_boxes_num - labels.shape[0]), value=-1))
            if boxes.shape[0] != 0:
                pad_boxes.append(
                    F.pad(boxes, (0, 0, 0, max_boxes_num - boxes.shape[0]),
                          value=-1))
            else:
                boxes.unsqueeze_(0)
                pad_boxes.append(
                    F.pad(boxes, (0, 4, 0, max_boxes_num - boxes.shape[0]),
                          value=-1))

        batch_imgs = torch.stack(pad_imgs, dim=0)
        batch_labels = torch.stack(pad_labels, dim=0)
        batch_boxes = torch.stack(pad_boxes, dim=0)

        return batch_imgs, batch_labels, batch_boxes


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

    print(labels, boxes)
    print(imgs.shape, boxes.shape, labels.shape)
