# -*- coding: utf-8 -*-
"""
# @file name  : transform.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-20
# @brief      : 数据变换
"""

import math
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img, boxes=None):
        norm_img = (img.astype(np.float32) / 255.0 - self.mean) / self.std

        if boxes is None:
            return norm_img
        else:
            return norm_img, boxes


class Colorjitter:
    def __init__(self,
                 brightness=0.1,
                 contrast=0.1,
                 saturation=0.1,
                 hue=0.1,
                 p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = p

    def __call__(self, img, boxes):
        if random.random() < self.prob:
            img = Image.fromarray(img)
            img = transforms.ColorJitter(self.brightness, self.contrast,
                                         self.saturation, self.hue)(img)
            cj_img = np.asarray(img, dtype=np.uint8)

            return cj_img, boxes
        else:
            return img, boxes


class Resize:
    def __init__(self, size=[800, 1333]):
        self.size = size
        self.min_size = min(size)
        self.max_size = max(size)

    def __call__(self, img, boxes=None):
        h, w, c = img.shape
        min_side, max_side = min(w, h), max(w, h)

        scale = self.min_size / min_side
        if max_side * scale > self.max_size:
            scale = self.max_size / max_side

        nw, nh = int(w * scale), int(h * scale)
        resize_img = cv2.resize(img, (nw, nh))

        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32

        pad_img = np.zeros((nh + pad_h, nw + pad_w, c), dtype=np.uint8)
        pad_img[:nh, :nw, :] = resize_img

        if boxes is None:
            return pad_img

        if boxes.shape[0] != 0:
            boxes[..., [0, 2]] = boxes[..., [0, 2]] * scale
            boxes[..., [1, 3]] = boxes[..., [1, 3]] * scale

        return pad_img, boxes


class Flip:
    def __init__(self, p=0.5):
        self.prob = p

    def __call__(self, img, boxes):
        if random.random() < self.prob:
            w = img.shape[1]
            flip_img = cv2.flip(img, 1)

            if boxes.shape[0] != 0:
                boxes[..., [0, 2]] = w - boxes[..., [2, 0]]

            return flip_img, boxes
        else:
            return img, boxes


class Translate:
    def __init__(self, x=10, y=10, p=0.5):
        self.dx = x
        self.dy = y
        self.prob = p

    def __call__(self, img, boxes):
        if random.random() < self.prob:
            h, w = img.shape[:2]
            dx = random.randint(-self.dx, self.dx)
            dy = random.randint(-self.dy, self.dy)

            trans_img = np.zeros_like(img)

            if boxes.shape[0] != 0:
                if dx > 0 and dy > 0:
                    trans_img[dy:, dx:, :] = img[:h - dy, :w - dx, :]
                elif dx > 0 and dy <= 0:
                    trans_img[:h + dy, dx:, :] = img[-dy:, :w - dx, :]
                elif dx <= 0 and dy > 0:
                    trans_img[dy:, :w + dx, :] = img[:h - dy, -dx:, :]
                else:
                    trans_img[:h + dy, :w + dx, :] = img[-dy:, -dx:, :]

                boxes[..., [0, 2]] = (boxes[..., [0, 2]] + dx).clamp(min=0,
                                                                     max=w - 1)
                boxes[..., [1, 3]] = (boxes[..., [1, 3]] + dy).clamp(min=0,
                                                                     max=h - 1)

            return trans_img, boxes
        else:
            return img, boxes


class Rotate:
    def __init__(self, d=10, p=0.5):
        self.degree = d
        self.prob = p

    def __call__(self, img, boxes):
        if random.random() < self.prob:
            h, w = img.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            degree = random.uniform(-self.degree, self.degree)
            theta = -degree / 180.0 * math.pi

            rot_mat = cv2.getRotationMatrix2D((cx, cy), degree, 1)
            rot_img = cv2.warpAffine(img, rot_mat, (w, h))

            if boxes.shape[0] != 0:
                boxes = torch.from_numpy(boxes)
                rot_boxes = torch.zeros_like(boxes)
                rot_boxes[..., 0] = boxes[..., 1]
                rot_boxes[..., 1] = boxes[..., 0]
                rot_boxes[..., 2] = boxes[..., 3]
                rot_boxes[..., 3] = boxes[..., 2]

                for i in range(boxes.shape[0]):
                    ymin, xmin, ymax, xmax = rot_boxes[i, :]
                    x0, y0 = xmin, ymin
                    x1, y1 = xmin, ymax
                    x2, y2 = xmax, ymin
                    x3, y3 = xmax, ymax
                    pt = torch.tensor([
                        [y0, x0],
                        [y1, x1],
                        [y2, x2],
                        [y3, x3],
                    ],
                                      dtype=torch.float)

                    rot_pt = torch.zeros_like(pt)
                    rot_pt[:, 1] = (pt[:, 1] - cx) * math.cos(theta) - (
                        pt[:, 0] - cy) * math.sin(theta) + cx
                    rot_pt[:, 0] = (pt[:, 1] - cx) * math.sin(theta) + (
                        pt[:, 0] - cy) * math.cos(theta) + cy
                    ymax, xmax = rot_pt.max(dim=0)[0]
                    ymin, xmin = rot_pt.min(dim=0)[0]
                    rot_boxes[i] = torch.stack([ymin, xmin, ymax, xmax])

                rot_boxes[..., [1, 3]] = rot_boxes[...,
                                                   [1, 3]].clamp(min=0,
                                                                 max=w - 1)
                rot_boxes[..., [0, 2]] = rot_boxes[...,
                                                   [0, 2]].clamp(min=0,
                                                                 max=h - 1)
                boxes[..., 0] = rot_boxes[..., 1]
                boxes[..., 1] = rot_boxes[..., 0]
                boxes[..., 2] = rot_boxes[..., 3]
                boxes[..., 3] = rot_boxes[..., 2]
                boxes = boxes.numpy()

            return rot_img, boxes
        else:
            return img, boxes


class Compose:
    def __init__(self, trans_list):
        self.trans_list = trans_list

    def __call__(self, img, boxes):
        for trans in self.trans_list:
            img, boxes = trans(img, boxes)

        return img, boxes


class AugTransform:
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.trans = Compose([
            Resize(size),
            Flip(),
            Normalize(mean, std),
        ])

    def __call__(self, img, boxes):
        return self.trans(img, boxes)


class BaseTransform:
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.trans = Compose([
            Resize(size),
            Normalize(mean, std),
        ])

    def __call__(self, img, boxes):
        return self.trans(img, boxes)


if __name__ == "__main__":

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, ".."))

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    from bdd100k import BDD100KDataset
    from kitti import KITTIDataset

    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    size = [800, 1333]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_dir = os.path.join(BASE_DIR, "..", "data", "samples", "kitti")
    train_set = KITTIDataset(data_dir,
                             "training",
                             transform=AugTransform(size, mean, std))
    train_loader = DataLoader(train_set)

    for (img, labels, boxes) in train_loader:
        img = img.squeeze(0).data.numpy()  # bchw -> chw
        img = img.transpose((1, 2, 0))  # chw -> hwc
        for i in range(3):
            img[..., i] = (img[..., i] * std[i] + mean[i]) * 255.0
        img = img.astype(np.uint8)
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
