# -*- coding: utf-8 -*-
"""
# @file name  : target.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-09
# @brief      : FCOS训练目标类
"""

import torch
import torch.nn as nn

from config import FCOSConfig
from utils import coords2centers, coords2offsets, decode_coords, reshape_feat


class FCOSTarget(nn.Module):
    def __init__(self, cfg=None):
        super(FCOSTarget, self).__init__()
        if cfg is None:
            self.cfg = FCOSConfig
        else:
            self.cfg = cfg

        self.strides = self.cfg.strides
        self.ranges = self.cfg.ranges
        assert len(self.strides) == len(self.ranges)

    def forward(self, feats, cls_ids, boxes):
        stages_num = len(self.strides)
        assert len(feats) == stages_num

        cls_targets = []
        reg_targets = []
        ctr_targets = []

        for i in range(stages_num):
            stage_targets = self._gen_stage_targets(
                feats[i],
                cls_ids,
                boxes,
                self.strides[i],
                self.ranges[i],
            )
            cls_targets.append(stage_targets[0])
            reg_targets.append(stage_targets[1])
            ctr_targets.append(stage_targets[2])

        return cls_targets, reg_targets, ctr_targets

    def _gen_stage_targets(self,
                           feat,
                           cls_ids,
                           boxes,
                           stride,
                           range,
                           sample_radio=1.5):
        coords = decode_coords(feat, stride).to(device=boxes.device)
        feat = reshape_feat(feat)  # bchw -> b(hw)c

        batch_size, hw = feat.shape[:2]  # b(hw)c
        boxes_num = boxes.shape[1]  # bnc

        # 1.计算每个坐标到所有标注框四边的偏移量
        offsets = coords2offsets(coords, boxes)
        assert offsets.shape == (batch_size, hw, boxes_num, 4)

        offsets_min = offsets.min(dim=-1)[0]
        offsets_max = offsets.max(dim=-1)[0]
        boxes_mask = offsets_min > 0
        stage_mask = (offsets_max > range[0]) & (offsets_max <= range[1])

        # 2.计算每个坐标到所有标注框中心的偏移量
        ctr_offsets = coords2centers(coords, boxes)
        assert ctr_offsets.shape == (batch_size, hw, boxes_num, 4)

        radius = sample_radio * stride
        ctr_offsets_max = ctr_offsets.max(dim=-1)[0]
        ctr_mask = ctr_offsets_max <= radius

        pos_mask = boxes_mask & stage_mask & ctr_mask
        assert pos_mask.shape == (batch_size, hw, boxes_num)

        # 3.计算所有标注框面积
        areas = (offsets[..., 0] + offsets[..., 2]) * (offsets[..., 1] +
                                                       offsets[..., 3])
        areas[~pos_mask] = 99999999  # neg_areas
        areas_min_idx = areas.min(dim=-1)[1].unsqueeze(dim=-1)
        areas_min_mask = torch.zeros_like(areas, dtype=torch.bool).scatter(
            -1, areas_min_idx, 1)
        assert areas_min_mask.shape == (batch_size, hw, boxes_num)

        # 4.计算分类目标
        cls_ids = torch.broadcast_tensors(
            cls_ids[:, None, :], areas.long())[0]  # [b,1,n] -> [b,h*w,n]
        cls_targets = cls_ids[areas_min_mask].reshape((batch_size, -1, 1))
        assert cls_targets.shape == (batch_size, hw, 1)

        # 5.计算回归目标
        offsets = offsets / stride
        reg_targets = offsets[areas_min_mask].reshape((batch_size, -1, 4))
        assert reg_targets.shape == (batch_size, hw, 4)

        # 6.计算中心度目标
        lr_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])
        lr_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        tb_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        tb_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        ctr_targets = ((lr_min * tb_min) /
                       (lr_max * tb_max).clamp(min=1e-10)).sqrt().unsqueeze(
                           dim=-1)
        assert ctr_targets.shape == (batch_size, hw, 1)

        # 7.处理负样本
        pos_mask = pos_mask.sum(dim=-1).long()
        pos_mask = pos_mask >= 1
        assert pos_mask.shape == (batch_size, hw)

        cls_targets[~pos_mask] = 0
        reg_targets[~pos_mask] = -1
        ctr_targets[~pos_mask] = -1

        return cls_targets, reg_targets, ctr_targets


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    model = FCOSTarget()

    preds = (
        [torch.rand(2, 3, 2, 2)] * 5,
        [torch.rand(2, 4, 2, 2)] * 5,
        [torch.rand(2, 1, 2, 2)] * 5,
    )
    cls_ids = torch.rand(2, 3)
    boxes = torch.rand(2, 3, 4)

    out = model(preds[0], cls_ids, boxes)
    [print(stage_out.shape) for branch_out in out for stage_out in branch_out]
