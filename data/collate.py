# -*- coding: utf-8 -*-
"""
# @file name  : collate.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-19
# @brief      : 数据打包
"""

import torch
import torch.nn.functional as F


class Collate(object):
    def __call__(self, data):
        imgs, labels, boxes = zip(*data)
        assert len(imgs) == len(labels) == len(boxes)

        pad_imgs = []
        pad_labels = []
        pad_boxes = []

        max_height = max(tuple(map(lambda img: img.shape[1], imgs)))
        max_width = max(tuple(map(lambda img: img.shape[2], imgs)))
        max_num_boxes = max(tuple(map(lambda boxes: boxes.shape[0], boxes)))

        for img, label, box in zip(imgs, labels, boxes):
            pad_imgs.append(
                F.pad(img, (0, max_width - img.shape[2], 0,
                            max_height - img.shape[1]),
                      value=0.0))

            pad_labels.append(
                F.pad(label, (0, max_num_boxes - label.shape[0]), value=-1))

            if box.shape[0] == 0:
                box.unsqueeze_(dim=0)

            pad_boxes.append(
                F.pad(box,
                      (0, 4 - box.shape[1], 0, max_num_boxes - box.shape[0]),
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

    size = (800, 1333)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    data_folder = "kitti"
    # data_folder = "bdd100k"

    data_dir = os.path.join(BASE_DIR, "samples", data_folder)

    if data_folder == "kitti":
        train_set = KITTIDataset(data_dir,
                                 "training",
                                 transform=BaseTransform(size, mean, std))
    elif data_folder == "bdd100k":
        train_set = BDD100KDataset(data_dir,
                                   "train",
                                   transform=BaseTransform(size, mean, std))
    train_loader = DataLoader(train_set, batch_size=4, collate_fn=Collate())

    imgs, labels, boxes = next(iter(train_loader))

    print(labels), print(boxes)
    print(imgs.shape, boxes.shape, labels.shape)
