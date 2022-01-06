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
    use_ctr = True
