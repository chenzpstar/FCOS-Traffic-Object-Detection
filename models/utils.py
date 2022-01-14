# -*- coding: utf-8 -*-
"""
# @file name  : utils.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-13
# @brief      : FCOS工具
"""

import torch


def reshape_feats(feats):
    feats_reshape = [reshape_feat(feat) for feat in feats]

    return torch.cat(feats_reshape, dim=1)


def reshape_feat(feat):
    batch_size, chn_num = feat.shape[:2]  # bchw
    feat = feat.permute(0, 2, 3, 1)  # bchw -> bhwc
    feat = feat.reshape((batch_size, -1, chn_num))  # bhwc -> b(hw)c

    return feat


def decode_preds(preds, strides=None):
    boxes = []

    if strides is not None:
        for pred, stride in zip(preds, strides):
            coord = decode_coords(pred, stride).to(device=pred.device)
            box = coords2boxes(coord, pred, stride)
            boxes.append(box)
    else:
        for pred in preds:
            coord = decode_coords(pred).to(device=pred.device)
            box = coords2boxes(coord, pred)
            boxes.append(box)

    return torch.cat(boxes, dim=1)


def decode_targets(preds, targets):
    boxes = []
    for pred, target in zip(preds, targets):
        coord = decode_coords(pred).to(device=target.device)
        box = coords2boxes(coord, target)
        boxes.append(box)

    return torch.cat(boxes, dim=1)


def decode_coords(feat, stride=1):
    h, w = feat.shape[2:]  # bchw
    x_shifts = torch.arange(0, w * stride, stride, dtype=torch.float)
    y_shifts = torch.arange(0, h * stride, stride, dtype=torch.float)

    y_shift, x_shift = torch.meshgrid(y_shifts, x_shifts)
    x_shift = x_shift.reshape(-1)
    y_shift = y_shift.reshape(-1)

    return torch.stack([x_shift, y_shift], dim=-1) + stride // 2


def coords2boxes(coords, offsets, stride=1):
    if len(offsets.shape) == 4:
        offsets = reshape_feat(offsets)  # bchw -> b(hw)c

    offsets = offsets * stride
    # xy - lt -> xy1
    boxes_xy1 = coords[None, :, :] - offsets[..., :2]
    # xy + rb -> xy2
    boxes_xy2 = coords[None, :, :] + offsets[..., 2:]

    return torch.cat([boxes_xy1, boxes_xy2], dim=-1)


def coords2offsets(coords, boxes):
    # xy - xy1 -> lt
    lt_offsets = coords[None, :, None] - boxes[..., :2][:, None, :]
    # xy2 - xy -> rb
    rb_offsets = boxes[..., 2:][:, None, :] - coords[None, :, None]

    return torch.cat([lt_offsets, rb_offsets], dim=-1)


def coords2centers(coords, boxes):
    # (xy1 + xy2) / 2 -> cxy
    boxes_cxy = (boxes[..., :2] + boxes[..., 2:]) / 2
    # xy - cxy -> lt
    lt_ctr_offsets = coords[None, :, None] - boxes_cxy[:, None, :]
    # cxy - xy -> rb
    rb_ctr_offsets = -lt_ctr_offsets

    return torch.cat([lt_ctr_offsets, rb_ctr_offsets], dim=-1)


def box_nms(cls_scores, boxes, thr=0.6, mode="iou"):
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    assert boxes.shape[-1] == 4

    xy1, xy2 = boxes[:, :2], boxes[:, 2:]
    if mode == "diou":
        cxy = (xy1 + xy2) / 2
    wh = xy2 - xy1
    areas = wh[:, 0] * wh[:, 1]
    order = cls_scores.sort(dim=0, descending=True)[1]

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        order = order[1:]
        xy1_max = torch.max(xy1[i], xy1[order])
        xy2_min = torch.min(xy2[i], xy2[order])
        wh_min = (xy2_min - xy1_max).clamp(min=0)

        overlap = wh_min[:, 0] * wh_min[:, 1]
        union = areas[i] + areas[order] - overlap
        iou = overlap / union.clamp(min=1e-10)

        if mode == "diou":
            xy1_min = torch.min(xy1[i], xy1[order])
            xy2_max = torch.max(xy2[i], xy2[order])
            wh_max = xy2_max - xy1_min
            cwh = cxy[i] - cxy[order]

            c_dist = wh_max[:, 0].pow(2) + wh_max[:, 1].pow(2)
            p_dist = cwh[:, 0].pow(2) + cwh[:, 1].pow(2)
            iou -= p_dist / c_dist.clamp(min=1e-10)

        idx = (iou <= thr).nonzero().squeeze(dim=-1)
        if idx.numel() == 0:
            break
        order = order[idx]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
