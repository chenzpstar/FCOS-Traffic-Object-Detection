# -*- coding: utf-8 -*-
# kitti_config.py

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import torch
from data.transform import AugTransform, BaseTransform
from easydict import EasyDict

cfg = EasyDict()
cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg.data_root_dir = os.path.join(BASE_DIR, "..", "..", "datasets")
cfg.data_folder = "kitti"

cfg.ckpt_root_dir = os.path.join(BASE_DIR, "..", "..", "results")
cfg.ckpt_folder = None

# transform
cfg.size = [800, 1333]  # [800, 1333]
cfg.mean = [0.367, 0.386, 0.374]  # [0.485, 0.456, 0.406]
cfg.std = [0.316, 0.321, 0.326]  # [0.229, 0.224, 0.225]

cfg.base_tf = BaseTransform(cfg.size, cfg.mean, cfg.std)
cfg.aug_tf = AugTransform(cfg.size, cfg.mean, cfg.std)

# backbone
# cfg.backbone = "vgg16"  # ["vgg16", "resnet50", "darknet19", "mobilenet", "shufflenet", "efficientnet"]
cfg.backbone = "resnet50"
# cfg.backbone = "darknet19"
# cfg.backbone = "mobilenet"
# cfg.backbone = "shufflenet"
# cfg.backbone = "efficientnet"
cfg.pretrained = True
cfg.freeze_bn = False  # freeze bn's statistics
cfg.freeze_bn_affine = False  # freeze bn's params

# neck
cfg.neck = "fpn"  # ["fpn", "pan", "bifpn"]
# cfg.neck = "pan"
# cfg.neck = "bifpn"
cfg.num_channels = 256  # 256
cfg.use_p5 = True

# head
cfg.num_convs = 4  # 4
cfg.num_classes = 3  # 8
cfg.prior = 0.01  # 0.01
cfg.use_gn = True
cfg.ctr_on_reg = True

# target
cfg.strides = [8, 16, 32, 64, 128]
cfg.bounds = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 1e6]]
cfg.use_ctr_sampling = True
cfg.ctr_ratio = 1.5  # 1.5

# loss
# cfg.cls_loss = "bce"  # ["bce", "focal"]
cfg.cls_loss = "focal"
# cfg.reg_loss = "smooth_l1"  # ["smooth_l1", "iou", "giou", "diou", "ciou"]
cfg.reg_loss = "iou"
# cfg.reg_loss = "giou"
# cfg.reg_loss = "diou"
# cfg.reg_loss = "ciou"
cfg.use_ctrness = True

# detect
cfg.max_num_boxes = 150  # 1000
cfg.score_thr = 0.05  # 0.05
cfg.nms_iou_thr = 0.5  # 0.6
cfg.nms_mode = "iou"  # ["iou", "diou"]
# cfg.nms_mode = "diou"

# dataloader
cfg.train_bs = 4
cfg.valid_bs = 4
cfg.workers = 4

# optimizer
cfg.lr_init = 0.005  # 0.01
cfg.momentum = 0.9  # 0.9
cfg.weight_decay = 5e-4  # 1e-4

# scheduler
cfg.factor = 0.1
cfg.milestones = [8, 11]
cfg.max_epoch = 12

cfg.exp_lr = False
cfg.exp_factor = 0.66  # 0.66^11 = 0.01

cfg.cos_lr = False
cfg.lr_final = 5e-5

cfg.warmup = True
cfg.warmup_factor = 0.001

# other
cfg.log_interval = 100
cfg.use_fp16 = False
cfg.clip_grad = False
cfg.use_07_metric = False
