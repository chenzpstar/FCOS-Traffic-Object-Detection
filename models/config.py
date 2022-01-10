# -*- coding: utf-8 -*-
"""
# @file name  : config.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-05
# @brief      : FCOS配置
"""


class FCOSConfig():
    # backbone
    pretrained = False

    # neck
    num_feat = 256
    use_p5 = True

    # head
    num_cls = 3
    use_gn = True
    ctr_on_reg = True
    prior = 0.01

    # loss
    cls_loss = "focal"
    reg_loss = "giou"
    use_ctr = True

    # target
    strides = [8, 16, 32, 64, 128]
    ranges = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 99999999]]

    # nms
    score_thr = 0.05
    nms_iou_thr = 0.5
    max_boxes_num = 150
