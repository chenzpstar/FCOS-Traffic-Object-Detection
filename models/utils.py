# -*- coding: utf-8 -*-
"""
# @file name  : utils.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-13
# @brief      : FCOS工具
"""

import torch

__all__ = [
    'reshape_feats', 'reshape_feat', 'decode_preds', 'decode_targets',
    'decode_coords', 'coords2boxes', 'coords2offsets', 'coords2centers',
    'nms_boxes', 'clip_boxes'
]


def reshape_feats(feats):
    out = list(map(reshape_feat, feats))

    return torch.cat(out, dim=1)


def reshape_feat(feat):
    b, c = feat.shape[:2]  # bchw
    feat = feat.permute(0, 2, 3, 1).reshape((b, -1, c))  # bchw -> b(hw)c

    return feat


def decode_preds(preds, strides=None):
    boxes = []

    if strides is not None:
        for pred, stride in zip(preds, strides):
            coord = decode_coords(pred, stride).to(pred.device)
            box = coords2boxes(coord, pred, stride)
            boxes.append(box)
    else:
        for pred in preds:
            coord = decode_coords(pred).to(pred.device)
            box = coords2boxes(coord, pred)
            boxes.append(box)

    return torch.cat(boxes, dim=1)


def decode_targets(preds, targets):
    boxes = []

    for pred, target in zip(preds, targets):
        coord = decode_coords(pred).to(target.device)
        box = coords2boxes(coord, target)
        boxes.append(box)

    return torch.cat(boxes, dim=1)


def decode_coords(feat, stride=1):
    h, w = feat.shape[-2:]  # bchw
    x_shifts = (torch.arange(0, w) + 0.5) * stride
    y_shifts = (torch.arange(0, h) + 0.5) * stride

    y_shift, x_shift = torch.meshgrid(y_shifts, x_shifts)
    x_shift = x_shift.reshape(-1)
    y_shift = y_shift.reshape(-1)

    return torch.stack([x_shift, y_shift], dim=-1)


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
    boxes_cxy = (boxes[..., :2] + boxes[..., 2:]) / 2.0
    # xy - cxy -> lt
    lt_ctr_offsets = coords[None, :, None] - boxes_cxy[:, None, :]
    # cxy - xy -> rb
    rb_ctr_offsets = -lt_ctr_offsets

    return torch.cat([lt_ctr_offsets, rb_ctr_offsets], dim=-1)


def nms_boxes(scores, boxes, iou_thr=0.5, mode="iou"):
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long)
    assert boxes.shape[-1] == 4

    xy1, xy2 = boxes[:, :2], boxes[:, 2:]
    if mode == "diou":
        cxy = (xy1 + xy2) / 2.0
    wh = xy2 - xy1
    areas = wh[:, 0] * wh[:, 1]
    order = scores.sort(dim=0, descending=True)[1]

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
        wh_min = (xy2_min - xy1_max).clamp_(min=0)

        overlap = wh_min[:, 0] * wh_min[:, 1]
        union = areas[i] + areas[order] - overlap
        iou = overlap / union.clamp_(min=1e-8)

        if mode == "diou":
            xy1_min = torch.min(xy1[i], xy1[order])
            xy2_max = torch.max(xy2[i], xy2[order])
            wh_max = xy2_max - xy1_min
            cwh = cxy[i] - cxy[order]

            c_dist = wh_max[:, 0].pow(2) + wh_max[:, 1].pow(2)
            p_dist = cwh[:, 0].pow(2) + cwh[:, 1].pow(2)
            iou -= p_dist / c_dist.clamp_(min=1e-8)

        idx = torch.where(iou <= iou_thr)[0]
        if idx.numel() == 0:
            break
        order = order[idx]

    return torch.LongTensor(keep)


def clip_boxes(img, boxes):
    h, w = img.shape[-2:]  # chw
    boxes.clamp_(min=0)
    boxes[..., [0, 2]].clamp_(max=w - 1)
    boxes[..., [1, 3]].clamp_(max=h - 1)

    return boxes
