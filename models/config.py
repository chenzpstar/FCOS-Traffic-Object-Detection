# -*- coding: utf-8 -*-
"""
# @file name  : config.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-05
# @brief      : FCOS配置类
"""


class FCOSConfig():
    # backbone
    backbone = "resnet50"  # ["vgg16", "resnet50", "darknet19"]
    pretrained = True

    # neck
    neck = "fpn"  # ["fpn, pan"]
    num_channels = 256  # 256
    use_p5 = True

    # head
    num_convs = 4  # 4
    num_classes = 3
    use_gn = True
    ctr_on_reg = True
    prior = 0.01  # 0.01

    # loss
    cls_loss = "focal"  # ["bce", "focal"]
    reg_loss = "iou"  # ["smooth_l1", "iou", "giou", "diou"]
    use_ctr = True

    # target
    strides = [8, 16, 32, 64, 128]
    ranges = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

    # detect
    max_boxes_num = 150  # 1000
    score_thr = 0.05  # 0.05
    nms_iou_thr = 0.5  # 0.6
    nms_mode = "iou"  # ["iou", "diou"]
