# -*- coding: utf-8 -*-
# kitti_config.py

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import torch
from data.transform import TestTransform, TrainTransform
from easydict import EasyDict

cfg = EasyDict()
cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg.data_root_dir = os.path.join(BASE_DIR, "..", "..", "datasets")
cfg.data_folder = "kitti"

cfg.ckpt_root_dir = os.path.join(BASE_DIR, "..", "..", "results")
cfg.ckpt_folder = None

# backbone
cfg.backbone = "resnet50"  # ["resnet50", "darknet19", "vgg16"]
# cfg.backbone = "darknet19"
cfg.pretrained = True

# neck
cfg.neck = "fpn"  # ["fpn, pan"]
# cfg.neck = "pan"
cfg.num_feat = 256  # 256
cfg.use_p5 = True

# head
cfg.num_cls = 3
cfg.use_gn = True
cfg.ctr_on_reg = True
cfg.prior = 0.01  # 0.01

# loss
cfg.cls_loss = "bce"  # ["bce", "focal"]
# cfg.cls_loss = "focal"
cfg.reg_loss = "iou"  # ["smooth_l1", "iou", "giou", "diou"]
# cfg.reg_loss = "giou"
# cfg.reg_loss = "diou"
cfg.use_ctr = True

# target
cfg.strides = [8, 16, 32, 64, 128]
cfg.ranges = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

# detect
cfg.score_thr = 0.05  # 0.05
cfg.nms_iou_thr = 0.5  # 0.6
cfg.max_boxes_num = 150  # 1000
cfg.nms_mode = "iou"  # ["iou", "diou"]
# cfg.nms_mode = "diou"

# dataloader
cfg.train_bs = 8
cfg.valid_bs = 4
cfg.workers = 16

# optimizer
cfg.lr_init = 0.01
cfg.momentum = 0.9
cfg.weight_decay = 1e-4

# scheduler
cfg.exp_lr = False
cfg.exp_factor = 0.98

cfg.factor = 0.1
cfg.milestones = [8, 11]
cfg.max_epoch = 12

# transform
cfg.size = [800, 1333]
cfg.mean = [0.485, 0.456, 0.406]
cfg.std = [0.229, 0.224, 0.225]

cfg.train_trans = TrainTransform(cfg.size, cfg.mean, cfg.std)
cfg.valid_trans = TestTransform(cfg.size, cfg.mean, cfg.std)
