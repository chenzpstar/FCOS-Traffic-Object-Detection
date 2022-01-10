# -*- coding: utf-8 -*-
"""
# @file name  : target.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-09
# @brief      : FCOS训练目标
"""

import torch
import torch.nn as nn

from config import FCOSConfig


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

    def forward(self, preds, boxes, cls_ids):
        cls_targets = []
        reg_targets = []
        ctr_targets = []

        cls_logits, reg_preds, ctr_logits = preds
        assert len(self.strides) == len(cls_logits)

        for stage in range(len(cls_logits)):
            stage_out = [
                cls_logits[stage], reg_preds[stage], ctr_logits[stage]
            ]
            stage_targets = self._gen_stage_targets(
                stage,
                stage_out,
                boxes,
                cls_ids,
            )
            cls_targets.append(stage_targets[0])
            reg_targets.append(stage_targets[1])
            ctr_targets.append(stage_targets[2])

        return (
            torch.cat(cls_targets, dim=1),
            torch.cat(reg_targets, dim=1),
            torch.cat(ctr_targets, dim=1),
        )

    def _gen_stage_targets(self,
                           stage,
                           stage_out,
                           boxes,
                           cls_ids,
                           sample_radio=1.5):
        cls_logits, reg_preds, ctr_logits = stage_out
        batch_size, cls_num = cls_logits.shape[:2]
        boxes_num = boxes.shape[1]

        cls_logits = cls_logits.permute(0, 2, 3, 1)  # bchw -> bhwc
        coords = self.orig_coords(cls_logits,
                                  self.strides[stage]).to(device=boxes.device)
        cls_logits = cls_logits.reshape(
            (batch_size, -1, cls_num))  # bhwc -> b(hw)c
        reg_preds = reg_preds.permute(0, 2, 3, 1)  # bchw -> bhwc
        reg_preds = reg_preds.reshape((batch_size, -1, 4))  # bhwc -> b(hw)c
        ctr_logits = ctr_logits.permute(0, 2, 3, 1)  # bchw -> bhwc
        ctr_logits = ctr_logits.reshape((batch_size, -1, 1))  # bhwc -> b(hw)c

        hw = cls_logits.shape[1]

        # 1.计算每个坐标到所有标注框四边的偏移量
        x = coords[:, 0]
        y = coords[:, 1]
        l_offsets = x[None, :, None] - boxes[..., 0][:, None, :]
        t_offsets = y[None, :, None] - boxes[..., 1][:, None, :]
        r_offsets = boxes[..., 2][:, None, :] - x[None, :, None]
        b_offsets = boxes[..., 3][:, None, :] - y[None, :, None]
        offsets = torch.stack([l_offsets, t_offsets, r_offsets, b_offsets],
                              dim=-1)
        assert offsets.shape == (batch_size, hw, boxes_num, 4)

        offsets_min = offsets.min(dim=-1)[0]
        offsets_max = offsets.max(dim=-1)[0]
        boxes_mask = offsets_min > 0
        stage_mask = (offsets_max > self.ranges[stage][0]) & (
            offsets_max <= self.ranges[stage][1])

        # 2.计算每个坐标到所有标注框中心的偏移量
        boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
        boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
        l_ctr_offsets = x[None, :, None] - boxes_cx[:, None, :]
        t_ctr_offsets = y[None, :, None] - boxes_cy[:, None, :]
        r_ctr_offsets = -l_ctr_offsets
        b_ctr_offsets = -t_ctr_offsets
        ctr_offsets = torch.stack(
            [l_ctr_offsets, t_ctr_offsets, r_ctr_offsets, b_ctr_offsets],
            dim=-1)
        assert ctr_offsets.shape == (batch_size, hw, boxes_num, 4)

        radius = self.strides[stage] * sample_radio
        ctr_offsets_max = ctr_offsets.max(dim=-1)[0]
        ctr_mask = ctr_offsets_max < radius

        pos_mask = boxes_mask & stage_mask & ctr_mask
        assert pos_mask.shape == (batch_size, hw, boxes_num)

        # 3.计算所有标注框面积
        areas = (offsets[..., 0] + offsets[..., 2]) * (offsets[..., 1] +
                                                       offsets[..., 3])
        areas[~pos_mask] = 99999999  # neg_areas
        areas_min_idx = areas.min(dim=-1)[1].unsqueeze(dim=-1)
        areas_min_mask = torch.zeros_like(areas.bool()).scatter(
            -1, areas_min_idx, 1)
        assert areas_min_mask.shape == (batch_size, hw, boxes_num)

        # 4.计算分类目标
        cls_ids = torch.broadcast_tensors(cls_ids[:, None, :], areas.long())[0]
        cls_targets = cls_ids[areas_min_mask].reshape((batch_size, -1, 1))
        assert cls_targets.shape == (batch_size, hw, 1)

        # 5.计算回归目标
        offsets = offsets / self.strides[stage]
        reg_targets = offsets[areas_min_mask].reshape((batch_size, -1, 4))
        assert reg_targets.shape == (batch_size, hw, 4)

        # 6.计算中心度目标
        lr_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])
        lr_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        tb_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        tb_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        ctr_targets = ((lr_min * tb_min) /
                       (lr_max * tb_max).clamp(min=1e-10)).sqrt().unsqueeze(
                           dim=-1)  # b(hw) -> b(hw)c
        assert ctr_targets.shape == (batch_size, hw, 1)

        # 7.处理负样本
        pos_mask = pos_mask.long().sum(dim=-1)
        pos_mask = pos_mask >= 1
        assert pos_mask.shape == (batch_size, hw)

        cls_targets[~pos_mask] = 0
        reg_targets[~pos_mask] = -1
        ctr_targets[~pos_mask] = -1

        return cls_targets, reg_targets, ctr_targets

    @staticmethod
    def orig_coords(feat, stride):
        h, w = feat.shape[1:3]
        x_shifts = torch.arange(0, w * stride, stride, dtype=float)
        y_shifts = torch.arange(0, h * stride, stride, dtype=float)

        y_shift, x_shift = torch.meshgrid(y_shifts, x_shifts)
        x_shift = x_shift.reshape(-1)
        y_shift = y_shift.reshape(-1)

        coords = torch.stack([x_shift, y_shift], dim=-1) + stride // 2

        return coords


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    model = FCOSTarget()

    preds = [
        [torch.rand(2, 3, 2, 2)] * 5,
        [torch.rand(2, 4, 2, 2)] * 5,
        [torch.rand(2, 1, 2, 2)] * 5,
    ]
    boxes = torch.rand(2, 3, 4)
    cls_ids = torch.rand(2, 3)

    out = model(preds, boxes, cls_ids)
    [print(branch_out) for branch_out in out]
