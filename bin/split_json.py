# -*- coding: utf-8 -*-
"""
# @file name  : split_json.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-03
# @brief      : 切分json标注文件
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

from tqdm import tqdm


def split_json_file(file_dir):
    for file in os.listdir(file_dir):
        if file.endswith(".json"):
            json_path = os.path.join(file_dir, file)

            with open(json_path, "r") as json_file:
                annos = json.load(json_file)

            print("imgs num: {}".format(len(annos)))

            out_dir = json_path.replace(".json", "")
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            for anno in tqdm(annos):
                out_path = os.path.join(out_dir,
                                        anno["name"].replace(".jpg", ".json"))

                with open(out_path, "w") as out_file:
                    json.dump(anno, out_file, indent=4)


if __name__ == "__main__":

    root_dir = os.path.join(BASE_DIR, "..", "..", "datasets")
    json_dir = os.path.join(root_dir, "bdd100k", "labels", "100k")
    split_json_file(json_dir)
    print("splitting json files done")
