# -*- coding: utf-8 -*-
"""
# @file name  : loss.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-07
# @brief      : FCOS损失函数类
"""

from math import pi

import torch
import torch.nn as nn

try:
    from .config import FCOSConfig
    from .utils import (box_area, box_ratio, decode_preds, decode_targets,
                        offset_area, reshape_feats)
except:
    from config import FCOSConfig
    from utils import (box_area, box_ratio, decode_preds, decode_targets,
                       offset_area, reshape_feats)


def bce_loss(logits, targets, eps=1e-8):
    probs = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)
    loss = -(targets * torch.log(probs) +
             (1.0 - targets) * torch.log(1.0 - probs))

    return loss.sum()


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, eps=1e-8):
    probs = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)
    loss = -(alpha * (1.0 - probs).pow(gamma) * targets * torch.log(probs) +
             (1.0 - alpha) * probs.pow(gamma) *
             (1.0 - targets) * torch.log(1.0 - probs))

    return loss.sum()


def smooth_l1_loss(preds, targets):
    res = torch.abs(preds - targets)
    loss = torch.where(res < 1, 0.5 * res.pow(2), res - 0.5)

    return loss.sum()


def offset_iou_loss(preds, targets, mode="iou", eps=1e-8):
    # [l,t,r,b]
    lt_min = torch.min(preds[..., :2], targets[..., :2])
    rb_min = torch.min(preds[..., 2:], targets[..., 2:])
    wh_min = (lt_min + rb_min).clamp_(min=0)
    overlap = wh_min[..., 0] * wh_min[..., 1]

    union = offset_area(preds) + offset_area(targets) - overlap
    iou = overlap / union.clamp_(min=eps)

    if mode == "giou":
        lt_max = torch.max(preds[..., :2], targets[..., :2])
        rb_max = torch.max(preds[..., 2:], targets[..., 2:])
        wh_max = (lt_max + rb_max).clamp_(min=0)
        C_area = wh_max[..., 0] * wh_max[..., 1]

        iou -= (C_area - union) / C_area.clamp_(min=eps)

    loss = 1.0 - iou

    return loss.sum()


def box_iou_loss(preds, targets, mode="iou", eps=1e-8):
    # [x1,y1,x2,y2]
    xy1_max = torch.max(preds[..., :2], targets[..., :2])
    xy2_min = torch.min(preds[..., 2:], targets[..., 2:])
    wh_min = (xy2_min - xy1_max).clamp_(min=0)
    overlap = wh_min[..., 0] * wh_min[..., 1]

    union = box_area(preds) + box_area(targets) - overlap
    iou = overlap / union.clamp_(min=eps)

    if mode in ["giou", "diou", "ciou"]:
        xy1_min = torch.min(preds[..., :2], targets[..., :2])
        xy2_max = torch.max(preds[..., 2:], targets[..., 2:])
        wh_max = (xy2_max - xy1_min).clamp_(min=0)

        if mode == "giou":
            C_area = wh_max[..., 0] * wh_max[..., 1]

            iou -= (C_area - union) / C_area.clamp_(min=1e-8)
        else:
            c_dist = wh_max[..., 0].pow(2) + wh_max[..., 1].pow(2)

            pred_cxy = (preds[..., :2] + preds[..., 2:]) / 2.0
            target_cxy = (targets[..., :2] + targets[..., 2:]) / 2.0
            cwh = pred_cxy - target_cxy
            p_dist = cwh[..., 0].pow(2) + cwh[..., 1].pow(2)

            if mode == "diou":
                iou -= p_dist / c_dist.clamp_(min=eps)
            else:
                v = (4 / pi**2) * (torch.atan(box_ratio(targets)) -
                                   torch.atan(box_ratio(preds))).pow(2)
                with torch.no_grad():
                    alpha = v / (1.0 - iou + v).clamp_(min=eps)

                iou -= p_dist / c_dist.clamp_(min=eps) + alpha * v

    loss = 1.0 - iou

    return loss.sum()


def calc_cls_loss(logits, targets, mode="focal"):
    # logits: [b,h*w,c]
    # targets: [b,h*w,1]
    num_classes = logits.shape[-1]
    assert logits.shape[:2] == targets.shape[:2]

    loss = []
    for pos_logit, pos_target in zip(logits, targets):
        pos_label = torch.arange(1, num_classes + 1, device=pos_target.device)
        pos_target = (pos_target == pos_label[None, :]).float()  # one-hot
        assert pos_logit.shape == pos_target.shape

        if mode == "bce":
            loss.append(bce_loss(pos_logit, pos_target))
        elif mode == "focal":
            loss.append(focal_loss(pos_logit, pos_target))
        else:
            raise NotImplementedError(
                "cls loss only implemented ['bce', 'focal']")

    return torch.stack(loss, dim=0)


def calc_reg_loss(preds, targets, pos_masks, mode="iou"):
    # preds: [b,h*w,4]
    # targets: [b,h*w,4]
    # pos_masks: [b,h*w]
    assert preds.shape == targets.shape

    loss = []
    for pred, target, pos_mask in zip(preds, targets, pos_masks):
        pos_pred = pred[pos_mask]
        pos_target = target[pos_mask]
        assert pos_pred.shape == pos_target.shape

        if mode == "smooth_l1":
            loss.append(smooth_l1_loss(pos_pred, pos_target))
        elif mode in ["iou", "giou"]:
            loss.append(offset_iou_loss(pos_pred, pos_target, mode))
        elif mode in ["diou", "ciou"]:
            loss.append(box_iou_loss(pos_pred, pos_target, mode))
        else:
            raise NotImplementedError(
                "reg loss only implemented ['smooth_l1', 'iou', 'giou', 'diou', 'ciou]"
            )

    return torch.stack(loss, dim=0)


def calc_ctr_loss(logits, targets, pos_masks):
    # logits: [b,h*w,1]
    # targets: [b,h*w,1]
    # pos_masks: [b,h*w]
    assert logits.shape == targets.shape

    loss = []
    for logit, target, pos_mask in zip(logits, targets, pos_masks):
        pos_logit = logit[pos_mask]
        pos_target = target[pos_mask]
        assert pos_logit.shape == pos_target.shape

        loss.append(bce_loss(pos_logit, pos_target))

    return torch.stack(loss, dim=0)


class FCOSLoss(nn.Module):
    def __init__(self, cfg=None):
        super(FCOSLoss, self).__init__()
        self.cfg = FCOSConfig if cfg is None else cfg
        self.cls_loss = self.cfg.cls_loss
        self.reg_loss = self.cfg.reg_loss
        self.use_ctr = self.cfg.use_ctr

    def forward(self, preds, targets):
        cls_logits, reg_preds, ctr_logits = preds
        cls_targets, reg_targets, ctr_targets = targets

        cls_logits = reshape_feats(cls_logits)  # bchw -> b(hw)c
        ctr_logits = reshape_feats(ctr_logits)  # bchw -> b(hw)c
        cls_targets = torch.cat(cls_targets, dim=1)
        ctr_targets = torch.cat(ctr_targets, dim=1)

        pos_mask = (ctr_targets > -1).squeeze_(dim=-1)  # b(hw)c -> b(hw)
        num_pos = pos_mask.float().sum(dim=-1).clamp_(min=1)

        cls_loss = calc_cls_loss(cls_logits, cls_targets,
                                 self.cls_loss) / num_pos
        ctr_loss = calc_ctr_loss(ctr_logits, ctr_targets, pos_mask) / num_pos

        if self.reg_loss in ["diou", "ciou"]:
            pred_boxes = decode_preds(reg_preds)  # bchw -> b(hw)c
            target_boxes = decode_targets(reg_preds, reg_targets)  # b(hw)c
            reg_loss = calc_reg_loss(pred_boxes, target_boxes, pos_mask,
                                     self.reg_loss) / num_pos
        else:
            pred_offsets = reshape_feats(reg_preds)  # bchw -> b(hw)c
            target_offsets = torch.cat(reg_targets, dim=1)
            reg_loss = calc_reg_loss(pred_offsets, target_offsets, pos_mask,
                                     self.reg_loss) / num_pos

        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        ctr_loss = ctr_loss.mean() if self.use_ctr else 0.0
        total_loss = cls_loss + reg_loss + ctr_loss

        return total_loss, cls_loss, reg_loss, ctr_loss


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    flag = 0
    # flag = 1
    # flag = 2

    if flag == 0:
        model = FCOSLoss()

        preds = (
            [torch.rand(2, 3, 2, 2)] * 5,
            [torch.rand(2, 4, 2, 2)] * 5,
            [torch.rand(2, 1, 2, 2)] * 5,
        )
        targets = (
            [torch.rand(2, 4, 1)] * 5,
            [torch.rand(2, 4, 4)] * 5,
            [torch.rand(2, 4, 1)] * 5,
        )

        out = model(preds, targets)
        [print(branch_out.item()) for branch_out in out]

    if flag == 1:
        preds = torch.rand(2, 3, 2, 2)
        targets = torch.rand(2, 3, 2, 2)

        loss1 = bce_loss(preds, targets)
        loss2 = nn.BCEWithLogitsLoss(reduction="sum")(preds, targets)
        print(loss1, loss2)

    if flag == 2:
        preds = torch.rand(2, 4, 2, 2)
        targets = torch.rand(2, 4, 2, 2)

        loss1 = smooth_l1_loss(preds, targets)
        loss2 = nn.SmoothL1Loss(reduction="sum")(preds, targets)
        print(loss1, loss2)
