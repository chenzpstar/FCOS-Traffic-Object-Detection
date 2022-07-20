# -*- coding: utf-8 -*-
"""
# @file name  : config.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-05
# @brief      : FCOS配置类
"""


class FCOSConfig(object):
    # backbone
    backbone = "resnet50"  # ["vgg16", "resnet50", "darknet19", "darknet53", "mobilenet", "shufflenet", "efficientnet"]
    pretrained = True
    freeze_bn = False  # freeze bn's statistics
    freeze_bn_affine = False  # freeze bn's params
    freeze_backbone = False

    # neck
    neck = "fpn"  # ["fpn", "pan", "bifpn"]
    num_channels = 256  # 256
    use_p5 = True

    # head
    num_convs = 4  # 4
    num_classes = 3
    prior = 0.01  # 0.01
    use_gn = True
    ctr_on_reg = True

    # target
    strides = [8, 16, 32, 64, 128]
    bounds = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 1e6]]
    ctr_sampling = True
    sample_radius = 1.5  # 1.5

    # loss
    cls_loss = "fl"  # ["bce", "fl", "qfl"]
    reg_loss = "iou"  # ["smooth_l1", "iou", "giou", "diou", "ciou"]
    use_ctrness = True
    label_smoothing = False
    smooth_eps = 0.001  # 0.1

    # detect
    max_num_boxes = 1000  # 1000
    score_thr = 0.05  # 0.05
    nms_iou_thr = 0.6  # 0.6
    nms_method = "iou"  # ["iou", "diou"]


if __name__ == "__main__":

    cfg = FCOSConfig
    print(cfg.backbone, cfg.neck, cfg.cls_loss, cfg.reg_loss)
