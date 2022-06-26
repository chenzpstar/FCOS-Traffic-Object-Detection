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
import torch.nn.functional as F

try:
    from .config import FCOSConfig
    from .utils import box_area, box_ratio, decode_boxes, offset_area
except:
    from config import FCOSConfig
    from utils import box_area, box_ratio, decode_boxes, offset_area


def bce_loss(logits, targets, reduction="sum"):
    # probs = torch.sigmoid(logits)
    # loss = -(targets * torch.log(probs) +
    #          (1.0 - targets) * torch.log(1.0 - probs))

    loss = F.binary_cross_entropy_with_logits(logits,
                                              targets,
                                              reduction="none")

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()

    return loss


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    probs = torch.sigmoid(logits)
    loss = bce_loss(logits, targets, reduction=None)

    probs_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    loss *= (1.0 - probs_t).pow(gamma)

    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss *= alpha_t

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()

    return loss


def smooth_l1_loss(preds, targets, reduction="sum"):
    res = torch.abs(preds - targets)
    loss = torch.where(res < 1, 0.5 * res.pow(2), res - 0.5)

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()

    return loss


def offset_iou_loss(preds, targets, mode="iou", reduction="sum", eps=1e-8):
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

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()

    return loss


def box_iou_loss(preds, targets, mode="iou", reduction="sum", eps=1e-8):
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

            iou -= (C_area - union) / C_area.clamp_(min=eps)
        else:
            c_dist = wh_max[..., 0].pow(2) + wh_max[..., 1].pow(2)

            pred_cxy = (preds[..., :2] + preds[..., 2:]) / 2.0
            target_cxy = (targets[..., :2] + targets[..., 2:]) / 2.0
            cwh = pred_cxy - target_cxy
            p_dist = cwh[..., 0].pow(2) + cwh[..., 1].pow(2)

            iou -= p_dist / c_dist.clamp_(min=eps)

            if mode == "ciou":
                v = (4 / pi**2) * (torch.atan(box_ratio(targets)) -
                                   torch.atan(box_ratio(preds))).pow(2)
                with torch.no_grad():
                    alpha = v / (1.0 - iou + v).clamp_(min=eps)

                iou -= alpha * v

    loss = 1.0 - iou

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()

    return loss


def calc_cls_loss(logits, targets, num_pos, mode="focal"):
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

    return torch.stack(loss, dim=0) / num_pos


def calc_reg_loss(preds, targets, pos_masks, num_pos, mode="iou"):
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

    return torch.stack(loss, dim=0) / num_pos


def calc_ctr_loss(logits, targets, pos_masks, num_pos):
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

    return torch.stack(loss, dim=0) / num_pos


class FCOSLoss(nn.Module):
    def __init__(self, cfg=None):
        super(FCOSLoss, self).__init__()
        self.cfg = FCOSConfig if cfg is None else cfg
        self.cls_loss = self.cfg.cls_loss
        self.reg_loss = self.cfg.reg_loss
        self.use_ctr = self.cfg.use_ctr

    def forward(self, preds, targets):
        cls_logits, reg_preds, ctr_logits, coords = preds
        cls_targets, reg_targets, ctr_targets = targets

        cls_logits = torch.cat(cls_logits, dim=1)
        ctr_logits = torch.cat(ctr_logits, dim=1)
        cls_targets = torch.cat(cls_targets, dim=1)
        ctr_targets = torch.cat(ctr_targets, dim=1)

        if self.reg_loss in ["diou", "ciou"]:
            reg_preds = decode_boxes(reg_preds, coords)
            reg_targets = decode_boxes(reg_targets, coords)
        else:
            reg_preds = torch.cat(reg_preds, dim=1)
            reg_targets = torch.cat(reg_targets, dim=1)

        pos_mask = (ctr_targets > -1).squeeze_(dim=-1)  # b(hw)c -> b(hw)
        num_pos = pos_mask.float().sum(dim=-1).clamp_(min=1)  # b(hw) -> b

        cls_loss = calc_cls_loss(cls_logits, cls_targets, num_pos,
                                 self.cls_loss).mean()
        reg_loss = calc_reg_loss(reg_preds, reg_targets, pos_mask, num_pos,
                                 self.reg_loss).mean()
        ctr_loss = calc_ctr_loss(ctr_logits, ctr_targets, pos_mask,
                                 num_pos).mean() if self.use_ctr else 0.0
        total_loss = cls_loss + reg_loss + ctr_loss

        return total_loss, cls_loss, reg_loss, ctr_loss


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    from torchvision.ops import sigmoid_focal_loss

    from utils import decode_coords

    flag = 0
    # flag = 1
    # flag = 2
    # flag = 3
    # flag = 4

    if flag == 0:
        model = FCOSLoss()

        preds = (
            [torch.rand(2, 4, 3)] * 5,
            [torch.rand(2, 4, 4)] * 5,
            [torch.rand(2, 4, 1)] * 5,
            [torch.rand(4, 2)] * 5,
        )
        targets = (
            [torch.randint(4, (2, 4, 1))] * 5,
            [torch.rand(2, 4, 4)] * 5,
            [torch.rand(2, 4, 1)] * 5,
        )

        out = model(preds, targets)
        [print(branch_out.item()) for branch_out in out]

    if flag == 1:
        preds = torch.rand(2, 4, 3)
        targets = torch.rand(2, 4, 3)

        loss1 = bce_loss(preds, targets)
        loss2 = F.binary_cross_entropy_with_logits(preds,
                                                   targets,
                                                   reduction="sum")
        print(loss1.item() == loss2.item())

    if flag == 2:
        preds = torch.rand(2, 4, 3)
        targets = torch.rand(2, 4, 3)

        loss1 = focal_loss(preds, targets)
        loss2 = sigmoid_focal_loss(preds, targets, reduction="sum")
        print(loss1.item() == loss2.item())

    if flag == 3:
        preds = torch.rand(2, 4, 4)
        targets = torch.rand(2, 4, 4)

        loss1 = smooth_l1_loss(preds, targets)
        loss2 = F.smooth_l1_loss(preds, targets, reduction="sum")
        print(loss1.item() == loss2.item())

    if flag == 4:
        preds = torch.rand(2, 4, 2, 2)
        targets = torch.rand(2, 4, 2, 2)

        coords = list(map(decode_coords, preds))

        pred_offsets = preds.permute(0, 2, 3, 1).reshape((2, -1, 4))
        target_offsets = targets.permute(0, 2, 3, 1).reshape((2, -1, 4))

        pred_boxes = decode_boxes(pred_offsets, coords)
        target_boxes = decode_boxes(target_offsets, coords)

        loss1 = offset_iou_loss(pred_offsets, target_offsets, "giou")
        loss2 = box_iou_loss(pred_boxes, target_boxes, "giou")
        print(loss1.item() == loss2.item())
