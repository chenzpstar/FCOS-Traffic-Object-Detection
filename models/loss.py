# -*- coding: utf-8 -*-
"""
# @file name  : loss.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-07
# @brief      : FCOS损失函数
"""

import torch
import torch.nn as nn

from config import FCOSConfig


def bce_loss(logits, targets, eps=1e-10):
    probs = logits.sigmoid().clamp(min=eps, max=1.0 - eps)
    loss = -(targets * probs.log() + (1.0 - targets) * (1.0 - probs).log())

    return loss.sum()


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, eps=1e-10):
    probs = logits.sigmoid().clamp(min=eps, max=1.0 - eps)
    loss = -(alpha * (1.0 - probs).pow(gamma) * targets * probs.log() +
             (1.0 - alpha) * probs.pow(gamma) * (1.0 - targets) *
             (1.0 - probs).log())

    return loss.sum()


def smooth_l1_loss(preds, targets):
    loss = (preds - targets).abs()
    mask = loss < 1
    loss[mask] = 0.5 * loss[mask].pow(2)
    loss[~mask] -= 0.5

    return loss.sum()


def iou_loss(preds, targets):
    # [l,t,r,b]
    lt_min = torch.min(preds[:, :2], targets[:, :2])
    rb_min = torch.min(preds[:, 2:], targets[:, 2:])
    wh_min = (lt_min + rb_min).clamp(min=0)
    overlap = wh_min[:, 0] * wh_min[:, 1]
    pred_area = (preds[:, 0] + preds[:, 2]) * (preds[:, 1] + preds[:, 3])
    target_area = (targets[:, 0] + targets[:, 2]) * (targets[:, 1] +
                                                     targets[:, 3])
    union = pred_area + target_area - overlap
    iou = overlap / union.clamp(min=1e-10)
    loss = 1.0 - iou

    return loss.sum()


def giou_loss(preds, targets):
    # [l,t,r,b]
    lt_min = torch.min(preds[:, :2], targets[:, :2])
    rb_min = torch.min(preds[:, 2:], targets[:, 2:])
    wh_min = (lt_min + rb_min).clamp(min=0)
    overlap = wh_min[:, 0] * wh_min[:, 1]
    pred_area = (preds[:, 0] + preds[:, 2]) * (preds[:, 1] + preds[:, 3])
    target_area = (targets[:, 0] + targets[:, 2]) * (targets[:, 1] +
                                                     targets[:, 3])
    union = pred_area + target_area - overlap
    iou = overlap / union.clamp(min=1e-10)

    lt_max = torch.max(preds[:, :2], targets[:, :2])
    rb_max = torch.max(preds[:, 2:], targets[:, 2:])
    wh_max = (lt_max + rb_max).clamp(min=0)
    C_area = wh_max[:, 0] * wh_max[:, 1]

    giou = iou - (C_area - union) / C_area.clamp(min=1e-10)
    loss = 1.0 - giou

    return loss.sum()


def cal_cls_loss(logits, targets, pos_mask, mode="focal"):
    # logits: [b,c,h,w] -> [b,h*w,c]
    # targets: [b,h*w,1]
    # pos_mask: [b,h*w]
    batch_size, cls_num = logits[0].shape[:2]
    pos_num = pos_mask.sum(dim=1).clamp(min=1).float()

    logits_reshape = []
    for logit in logits:
        logit = logit.permute(0, 2, 3, 1)  # bchw -> bhwc
        logit = logit.reshape((batch_size, -1, cls_num))  # bhwc -> b(hw)c
        logits_reshape.append(logit)
    logits = torch.cat(logits_reshape, dim=1)
    assert logits.shape[:2] == targets.shape[:2]

    loss = []
    for i in range(batch_size):
        pos_logit = logits[i]
        pos_target = targets[i]

        pos_label = torch.arange(
            1, cls_num + 1).unsqueeze(dim=0).to(device=pos_target.device)
        pos_target = (pos_label == pos_target).float()
        assert pos_logit.shape == pos_target.shape

        if mode == "bce":
            loss.append(bce_loss(pos_logit, pos_target).unsqueeze(dim=0))
        elif mode == "focal":
            loss.append(focal_loss(pos_logit, pos_target).unsqueeze(dim=0))
        else:
            raise NotImplementedError(
                "cls loss only implemented ['bce', 'focal']")

    return torch.cat(loss, dim=0) / pos_num.clamp(min=1e-10)


def cal_reg_loss(preds, targets, pos_mask, mode="giou"):
    # preds: [b,4,h,w] -> [b,h*w,4]
    # targets: [b,h*w,4]
    # pos_mask: [b,h*w]
    batch_size, chn_num = preds[0].shape[:2]
    pos_num = pos_mask.sum(dim=1).clamp(min=1).float()

    preds_reshape = []
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)  # bchw -> bhwc
        pred = pred.reshape((batch_size, -1, chn_num))  # bhwc -> b(hw)c
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape == targets.shape

    loss = []
    for i in range(batch_size):
        pos_pred = preds[i][pos_mask[i]]
        pos_target = targets[i][pos_mask[i]]
        assert pos_pred.shape == pos_target.shape

        if mode == "smooth_l1":
            loss.append(smooth_l1_loss(pos_pred, pos_target).unsqueeze(dim=0))
        elif mode == "iou":
            loss.append(iou_loss(pos_pred, pos_target).unsqueeze(dim=0))
        elif mode == "giou":
            loss.append(giou_loss(pos_pred, pos_target).unsqueeze(dim=0))
        else:
            raise NotImplementedError(
                "reg loss only implemented ['smooth_l1', 'iou', 'giou']")

    return torch.cat(loss, dim=0) / pos_num.clamp(min=1e-10)


def cal_ctr_loss(logits, targets, pos_mask):
    # logits: [b,1,h,w] -> [b,h*w,1]
    # targets: [b,h*w,1]
    # pos_mask: [b,h*w]
    batch_size, chn_num = logits[0].shape[:2]
    pos_num = pos_mask.sum(dim=1).clamp(min=1).float()

    logits_reshape = []
    for logit in logits:
        logit = logit.permute(0, 2, 3, 1)  # bchw -> bhwc
        logit = logit.reshape((batch_size, -1, chn_num))  # bhwc -> b(hw)c
        logits_reshape.append(logit)
    logits = torch.cat(logits_reshape, dim=1)
    assert logits.shape == targets.shape

    loss = []
    for i in range(batch_size):
        pos_logit = logits[i][pos_mask[i]]
        pos_target = targets[i][pos_mask[i]]
        assert pos_logit.shape == pos_target.shape

        loss.append(bce_loss(pos_logit, pos_target).unsqueeze(dim=0))

    return torch.cat(loss, dim=0) / pos_num.clamp(min=1e-10)


class FCOSLoss(nn.Module):
    def __init__(self, cfg=None):
        super(FCOSLoss, self).__init__()
        if cfg is None:
            self.cfg = FCOSConfig
        else:
            self.cfg = cfg

        self.cls_loss = self.cfg.cls_loss
        self.reg_loss = self.cfg.reg_loss
        self.use_ctr = self.cfg.use_ctr

    def forward(self, preds, targets):
        cls_logits, reg_preds, ctr_logits = preds
        cls_targets, reg_targets, ctr_targets = targets
        pos_mask = (ctr_targets > -1).squeeze(dim=-1)  # b(hw)c -> b(hw)

        cls_loss = cal_cls_loss(cls_logits, cls_targets, pos_mask,
                                self.cls_loss).mean()
        reg_loss = cal_reg_loss(reg_preds, reg_targets, pos_mask,
                                self.reg_loss).mean()
        ctr_loss = cal_ctr_loss(ctr_logits, ctr_targets, pos_mask).mean()

        flag = 1.0 if self.use_ctr else 0.0
        total_loss = cls_loss + reg_loss + ctr_loss * flag

        return cls_loss, reg_loss, ctr_loss, total_loss


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    flag = 0

    if flag == 0:
        model = FCOSLoss()

        preds = [
            [torch.rand(2, 3, 2, 2)] * 5,
            [torch.rand(2, 4, 2, 2)] * 5,
            [torch.rand(2, 1, 2, 2)] * 5,
        ]
        targets = [
            torch.rand(2, 20, 1),
            torch.rand(2, 20, 4),
            torch.rand(2, 20, 1),
        ]

        out = model(preds, targets)
        [print(branch_out) for branch_out in out]

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
