# -*- coding: utf-8 -*-
"""
# @file name  : utils.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-13
# @brief      : FCOS工具
"""

import torch

__all__ = [
    'reshape_feat', 'reshape_feats', 'decode_preds', 'decode_targets',
    'decode_coords', 'coords2boxes', 'coords2offsets', 'coords2centers',
    'box_ratio', 'box_area', 'box_iou', 'offset_area', 'offset_iou',
    'nms_boxes', 'clip_boxes'
]


def reshape_feat(feat):
    b, c = feat.shape[:2]  # bchw
    feat = feat.permute(0, 2, 3, 1).reshape((b, -1, c))  # bchw -> b(hw)c

    return feat


def reshape_feats(feats):
    out = list(map(reshape_feat, feats))

    return torch.cat(out, dim=1)


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
    offsets_lt = coords[None, :, None] - boxes[..., :2][:, None, :]
    # xy2 - xy -> rb
    offsets_rb = boxes[..., 2:][:, None, :] - coords[None, :, None]

    return torch.cat([offsets_lt, offsets_rb], dim=-1)


def coords2centers(coords, boxes):
    # (xy1 + xy2) / 2 -> cxy
    boxes_cxy = (boxes[..., :2] + boxes[..., 2:]) / 2.0
    # xy - cxy -> lt
    ctr_offsets_lt = coords[None, :, None] - boxes_cxy[:, None, :]
    # cxy - xy -> rb
    ctr_offsets_rb = -ctr_offsets_lt

    return torch.cat([ctr_offsets_lt, ctr_offsets_rb], dim=-1)


def box_ratio(boxes):
    # [x1,y1,x2,y2]
    return (boxes[..., 2] - boxes[..., 0]) / (boxes[..., 3] - boxes[..., 1])


def box_area(boxes):
    # [x1,y1,x2,y2]
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_iou(boxes1, boxes2):
    # [x1,y1,x2,y2]
    xy1 = torch.max(boxes1[..., :2], boxes2[..., :2])
    xy2 = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    wh = (xy2 - xy1).clamp_(min=0)
    overlap = wh[..., 0] * wh[..., 1]

    union = box_area(boxes1) + box_area(boxes2) - overlap

    return overlap / union.clamp_(min=1e-8)


def offset_area(boxes):
    # [l,t,r,b]
    return (boxes[..., 0] + boxes[..., 2]) * (boxes[..., 1] + boxes[..., 3])


def offset_iou(boxes1, boxes2):
    # [l,t,r,b]
    lt = torch.min(boxes1[..., :2], boxes2[..., :2])
    rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    wh = (lt + rb).clamp_(min=0)
    overlap = wh[..., 0] * wh[..., 1]

    union = offset_area(boxes1) + offset_area(boxes2) - overlap

    return overlap / union.clamp_(min=1e-8)


def nms_boxes(boxes, scores, iou_thr=0.5, mode="iou"):
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    assert boxes.shape[-1] == 4

    keep = []
    order = (-scores).argsort(dim=0)

    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        order = order[1:]

        top_boxes, other_boxes = boxes[i], boxes[order]
        iou = box_iou(top_boxes, other_boxes)

        if mode == "diou":
            xy1_min = torch.min(top_boxes[..., :2], other_boxes[..., :2])
            xy2_max = torch.max(top_boxes[..., 2:], other_boxes[..., 2:])
            wh_max = (xy2_max - xy1_min).clamp_(min=0)
            c_dist = wh_max[..., 0].pow(2) + wh_max[..., 1].pow(2)

            top_cxy = (top_boxes[..., :2] + top_boxes[..., 2:]) / 2.0
            other_cxy = (other_boxes[..., :2] + other_boxes[..., 2:]) / 2.0
            cwh = top_cxy - other_cxy
            p_dist = cwh[..., 0].pow(2) + cwh[..., 1].pow(2)

            iou -= p_dist / c_dist.clamp_(min=1e-8)

        idx = torch.where(iou <= iou_thr)[0]
        if idx.numel() == 0:
            break
        order = order[idx]

    return torch.LongTensor(keep)


def clip_boxes(boxes, img):
    h, w = img.shape[-2:]  # chw
    boxes[..., [0, 2]].clamp_(min=0, max=w - 1)
    boxes[..., [1, 3]].clamp_(min=0, max=h - 1)

    return boxes
