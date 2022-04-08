# -*- coding: utf-8 -*-
"""
# @file name  : remove_img.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-03
# @brief      : 移除无标注图像文件
"""

import os
import shutil


def remove_nolabel_img(img_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for img in os.listdir(img_dir):
        if img.endswith(".jpg"):
            img_path = os.path.join(img_dir, img)
            anno_path = img_path.replace("images",
                                         "labels").replace(".jpg", ".json")
            if not os.path.isfile(anno_path):
                shutil.move(img_path, out_dir)


if __name__ == "__main__":

    root_dir = "/home/vipuser/Documents/data/bdd100k/images/100k"
    img_dir = os.path.join(root_dir, "train")
    out_dir = os.path.join(root_dir, "train_nolabel")
    remove_nolabel_img(img_dir, out_dir)
    print("finish removing nolabel img")
