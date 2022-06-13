# -*- coding: utf-8 -*-
"""
# @file name  : target.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-09
# @brief      : FCOS训练目标类
"""

import torch
import torch.nn as nn

try:
    from .config import FCOSConfig
    from .utils import (coords2centers, coords2offsets, decode_coords,
                        offset_area, reshape_feat)
except:
    from config import FCOSConfig
    from utils import (coords2centers, coords2offsets, decode_coords,
                       offset_area, reshape_feat)


class FCOSTarget(nn.Module):
    def __init__(self, cfg=None):
        super(FCOSTarget, self).__init__()
        self.cfg = FCOSConfig if cfg is None else cfg
        self.strides = self.cfg.strides
        self.bounds = self.cfg.bounds
        assert len(self.strides) == len(self.bounds)

    def forward(self, feats, labels, boxes):
        assert len(self.strides) == len(feats)

        cls_targets = []
        reg_targets = []
        ctr_targets = []

        for feat, stride, bound in zip(feats, self.strides, self.bounds):
            stage_targets = self._gen_stage_targets(
                feat,
                labels,
                boxes,
                stride,
                bound,
            )
            cls_targets.append(stage_targets[0])
            reg_targets.append(stage_targets[1])
            ctr_targets.append(stage_targets[2])

        return cls_targets, reg_targets, ctr_targets

    def _gen_stage_targets(self,
                           feat,
                           labels,
                           boxes,
                           stride,
                           bound,
                           sample_ratio=1.5):
        coords = decode_coords(feat, stride).to(boxes.device)
        feat = reshape_feat(feat)  # bchw -> b(hw)c

        batch_size, hw = feat.shape[:2]  # b(hw)c
        num_boxes = boxes.shape[1]  # bnc

        # 1. 计算每个坐标到所有标注框四边的偏移量
        offsets = coords2offsets(coords, boxes)
        assert offsets.shape == (batch_size, hw, num_boxes, 4)

        min_offsets = offsets.min(dim=-1)[0]
        max_offsets = offsets.max(dim=-1)[0]
        boxes_mask = min_offsets > 0
        stage_mask = (max_offsets > bound[0]) & (max_offsets <= bound[1])

        # 2. 计算每个坐标到所有标注框中心的偏移量
        ctr_offsets = coords2centers(coords, boxes)
        assert ctr_offsets.shape == (batch_size, hw, num_boxes, 4)

        radius = sample_ratio * stride
        max_ctr_offsets = ctr_offsets.max(dim=-1)[0]
        ctr_mask = max_ctr_offsets < radius

        pos_mask = boxes_mask & stage_mask & ctr_mask
        assert pos_mask.shape == (batch_size, hw, num_boxes)

        # 3. 计算所有标注框面积
        areas = offset_area(offsets)
        areas[~pos_mask] = 99999999  # neg_areas
        min_areas_idx = areas.argmin(dim=-1, keepdim=True)
        min_areas_mask = torch.zeros_like(areas, dtype=torch.bool).scatter_(
            -1, min_areas_idx, 1)
        assert min_areas_mask.shape == (batch_size, hw, num_boxes)

        # 4. 计算分类目标
        labels = torch.broadcast_tensors(
            labels[:, None, :], areas.long())[0]  # [b,1,n] -> [b,h*w,n]
        cls_targets = labels[min_areas_mask].reshape((batch_size, -1, 1))
        assert cls_targets.shape == (batch_size, hw, 1)

        # 5. 计算回归目标
        offsets = offsets / stride
        reg_targets = offsets[min_areas_mask].reshape((batch_size, -1, 4))
        assert reg_targets.shape == (batch_size, hw, 4)

        # 6. 计算中心度目标
        lr_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])
        lr_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        tb_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        tb_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        ctr_targets = torch.sqrt(
            (lr_min * tb_min) /
            (lr_max * tb_max).clamp_(min=1e-8)).unsqueeze_(dim=-1)
        assert ctr_targets.shape == (batch_size, hw, 1)

        # 7. 处理负样本
        num_pos_mask = pos_mask.long().sum(dim=-1)
        neg_mask = num_pos_mask < 1
        assert neg_mask.shape == (batch_size, hw)

        cls_targets[neg_mask] = 0
        reg_targets[neg_mask] = -1
        ctr_targets[neg_mask] = -1

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
    labels = torch.rand(2, 3)
    boxes = torch.rand(2, 3, 4)

    out = model(preds[0], labels, boxes)
    [print(stage_out.shape) for branch_out in out for stage_out in branch_out]
