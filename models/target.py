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

    def forward(self, preds, cls_ids, boxes):
        cls_logits, reg_preds, ctr_logits = preds
        stages_num = len(self.strides)
        assert len(cls_logits) == stages_num

        cls_targets = []
        reg_targets = []
        ctr_targets = []

        for i in range(stages_num):
            stage_out = [cls_logits[i], reg_preds[i], ctr_logits[i]]
            stage_targets = self._gen_stage_targets(
                stage_out,
                cls_ids,
                boxes,
                self.strides[i],
                self.ranges[i],
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
                           preds,
                           cls_ids,
                           boxes,
                           stride,
                           range,
                           sample_radio=1.5):
        cls_logits, reg_preds, ctr_logits = preds
        batch_size, cls_num = cls_logits.shape[:2]  # bchw
        boxes_num = boxes.shape[1]  # bnc

        coords = self.decode_coords(cls_logits, stride).to(device=boxes.device)
        cls_logits = cls_logits.permute(0, 2, 3, 1)  # bchw -> bhwc
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
        # [1,h*w,1] - [b,1,n] -> [b,h*w,n]
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
        stage_mask = (offsets_max > range[0]) & (offsets_max <= range[1])

        # 2.计算每个坐标到所有标注框中心的偏移量
        boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
        boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
        # [1,h*w,1] - [b,1,n] -> [b,h*w,n]
        l_ctr_offsets = x[None, :, None] - boxes_cx[:, None, :]
        t_ctr_offsets = y[None, :, None] - boxes_cy[:, None, :]
        r_ctr_offsets = -l_ctr_offsets
        b_ctr_offsets = -t_ctr_offsets
        ctr_offsets = torch.stack(
            [l_ctr_offsets, t_ctr_offsets, r_ctr_offsets, b_ctr_offsets],
            dim=-1)
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

    @staticmethod
    def decode_coords(feat, stride):
        h, w = feat.shape[2:]  # bchw
        x_shifts = torch.arange(0, w * stride, stride, dtype=float)
        y_shifts = torch.arange(0, h * stride, stride, dtype=float)

        y_shift, x_shift = torch.meshgrid(y_shifts, x_shifts)
        x_shift = x_shift.reshape(-1)
        y_shift = y_shift.reshape(-1)

        return torch.stack([x_shift, y_shift], dim=-1) + stride // 2


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    model = FCOSTarget()

    preds = [
        [torch.rand(2, 3, 2, 2)] * 5,
        [torch.rand(2, 4, 2, 2)] * 5,
        [torch.rand(2, 1, 2, 2)] * 5,
    ]
    cls_ids = torch.rand(2, 3)
    boxes = torch.rand(2, 3, 4)

    out = model(preds, cls_ids, boxes)
    [print(branch_out.shape) for branch_out in out]
