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

        pad_imgs_list = []
        pad_labels_list = []
        pad_boxes_list = []

        h_list = [int(img.shape[1]) for img in imgs_list]
        w_list = [int(img.shape[2]) for img in imgs_list]
        num_list = [int(boxes.shape[0]) for boxes in boxes_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        max_num = np.array(num_list).max()

        for img, labels, boxes in zip(imgs_list, labels_list, boxes_list):
            pad_imgs_list.append(
                F.pad(img, (0, int(max_w - img.shape[2]), 0,
                            int(max_h - img.shape[1])),
                      value=0.))
            pad_labels_list.append(
                F.pad(labels, (0, max_num - labels.shape[0]), value=-1))
            pad_boxes_list.append(
                F.pad(boxes, (0, 0, 0, max_num - boxes.shape[0]), value=-1))

        batch_imgs = torch.stack(pad_imgs_list)
        batch_labels = torch.stack(pad_labels_list)
        batch_boxes = torch.stack(pad_boxes_list)

        return batch_imgs, batch_labels, batch_boxes


if __name__ == "__main__":

    import os

    from torch.utils.data import DataLoader

    from bdd100k import BDD100KDataset

    data_dir = os.path.join("data", "samples", "bdd100k")
    train_set = BDD100KDataset(data_dir, "train")
    train_loader = DataLoader(train_set, batch_size=4, collate_fn=Collate())

    imgs, labels, boxes = next(iter(train_loader))

    print(labels, boxes)
    print(imgs.shape, boxes.shape, labels.shape)
