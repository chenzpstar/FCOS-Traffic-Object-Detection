# -*- coding: utf-8 -*-
"""
# @file name  : calc_mean_std.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-04-04
# @brief      : 统计均值和标准差
"""

import torch
from tqdm import tqdm


def calc_mean_std(data_loader):
    img_mean_sigma, img_std_sigma = 0.0, 0.0

    for data in tqdm(data_loader):
        img = data[0].cuda()
        num_channels = img.shape[1]  # bchw
        img /= 255.0
        img = img.permute(0, 2, 3, 1).reshape(
            (-1, num_channels))  # bchw -> (bhw)c
        img_mean_sigma += img.mean(dim=0)
        img_std_sigma += img.pow(2).mean(dim=0)

    num_batches = len(data_loader)
    img_mean = img_mean_sigma / num_batches
    img_std = torch.sqrt((img_std_sigma / num_batches - img_mean.pow(2)))

    return img_mean, img_std


if __name__ == '__main__':

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, ".."))

    from data import BDD100KDataset, Collate, KITTIDataset
    from torch.utils.data import DataLoader

    data_folder = "kitti"
    # data_folder = "bdd100k"

    root_dir = os.path.join(BASE_DIR, "..", "..", "datasets")
    data_dir = os.path.join(root_dir, data_folder)

    if data_folder == "kitti":
        train_set = KITTIDataset(
            data_dir,
            set_name="training",
            mode="train",
            split=True,
        )
        valid_set = KITTIDataset(
            data_dir,
            set_name="training",
            mode="valid",
            split=True,
        )
    elif data_folder == "bdd100k":
        train_set = BDD100KDataset(
            data_dir,
            set_name="train",
        )
        valid_set = BDD100KDataset(
            data_dir,
            set_name="val",
        )

    train_loader = DataLoader(
        train_set,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=Collate(),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=Collate(),
    )

    train_mean, train_std = calc_mean_std(train_loader)
    valid_mean, valid_std = calc_mean_std(valid_loader)

    print("train: mean: {}, std: {}".format(train_mean, train_std))
    print("valid: mean: {}, std: {}".format(valid_mean, valid_std))
