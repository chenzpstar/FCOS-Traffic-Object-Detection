# -*- coding: utf-8 -*-
"""
# @file name  : utils.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-13
# @brief      : FCOS工具
"""

import torch

__all__ = [
    'decode_coords', 'reshape_feats', 'coords2offsets', 'coords2centers',
    'coords2boxes', 'decode_boxes', 'box_ratio', 'box_area', 'box_iou',
    'offset_area', 'offset_iou', 'clip_boxes', 'nms_boxes'
]


def decode_coords(feat, stride=1):
    h, w = feat.shape[-2:]  # bchw
    x_shifts = (torch.arange(0, w) + 0.5) * stride
    y_shifts = (torch.arange(0, h) + 0.5) * stride

    y_shift, x_shift = torch.meshgrid(y_shifts, x_shifts)
    x_shift = x_shift.reshape(-1)  # [h,w] -> [h*w]
    y_shift = y_shift.reshape(-1)  # [h,w] -> [h*w]

    return torch.stack((x_shift, y_shift), dim=-1)


def reshape_feats(feats):
    b, c = feats[0].shape[:2]  # bchw
    out = list(
        map(lambda feat: feat.permute(0, 2, 3, 1).reshape((b, -1, c)),
            feats))  # bchw -> b(hw)c

    return out


def coords2offsets(coords, boxes):
    # xy - xy1 -> lt
    offsets_lt = coords[None, :, None] - boxes[..., :2][:, None]
    # xy2 - xy -> rb
    offsets_rb = boxes[..., 2:][:, None] - coords[None, :, None]

    return torch.cat((offsets_lt, offsets_rb), dim=-1)


def coords2centers(coords, boxes):
    # (xy1 + xy2) / 2 -> cxy
    boxes_cxy = (boxes[..., :2] + boxes[..., 2:]) / 2.0
    # xy - cxy -> lt
    ctr_offsets_lt = coords[None, :, None] - boxes_cxy[:, None]
    # cxy - xy -> rb
    ctr_offsets_rb = -ctr_offsets_lt

    return torch.cat((ctr_offsets_lt, ctr_offsets_rb), dim=-1)


def coords2boxes(coords, offsets, stride=None):
    if stride is not None:
        offsets *= stride
    # xy - lt -> xy1
    boxes_xy1 = coords[None, :] - offsets[..., :2]
    # xy + rb -> xy2
    boxes_xy2 = coords[None, :] + offsets[..., 2:]

    return torch.cat((boxes_xy1, boxes_xy2), dim=-1)


def decode_boxes(offsets, coords, strides=None):
    if strides is not None:
        boxes = tuple(map(coords2boxes, coords, offsets, strides))
    else:
        boxes = tuple(map(coords2boxes, coords, offsets))

    return torch.cat(boxes, dim=1)


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

    return overlap / union.clamp(min=1e-7)


def offset_area(offsets):
    # [l,t,r,b]
    return (offsets[..., 0] + offsets[..., 2]) * (offsets[..., 1] +
                                                  offsets[..., 3])


def offset_iou(offsets1, offsets2):
    # [l,t,r,b]
    lt = torch.min(offsets1[..., :2], offsets2[..., :2])
    rb = torch.min(offsets1[..., 2:], offsets2[..., 2:])
    wh = (lt + rb).clamp_(min=0)
    overlap = wh[..., 0] * wh[..., 1]

    union = offset_area(offsets1) + offset_area(offsets2) - overlap

    return overlap / union.clamp(min=1e-7)


def clip_boxes(boxes, imgs):
    h, w = imgs.shape[-2:]  # bchw
    boxes[..., [0, 2]].clamp_(min=0, max=w - 1)
    boxes[..., [1, 3]].clamp_(min=0, max=h - 1)

    return boxes


def nms_boxes(boxes, scores, iou_thr=0.5, method="iou"):
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    assert boxes.shape[-1] == 4

    keep = []
    order = (-scores).argsort()

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

        if method == "diou":
            xy1_min = torch.min(top_boxes[..., :2], other_boxes[..., :2])
            xy2_max = torch.max(top_boxes[..., 2:], other_boxes[..., 2:])
            wh_max = (xy2_max - xy1_min).clamp_(min=0)
            c_dist = wh_max[..., 0].pow(2) + wh_max[..., 1].pow(2)

            top_cxy = (top_boxes[..., :2] + top_boxes[..., 2:]) / 2.0
            other_cxy = (other_boxes[..., :2] + other_boxes[..., 2:]) / 2.0
            cwh = top_cxy - other_cxy
            p_dist = cwh[..., 0].pow(2) + cwh[..., 1].pow(2)

            iou -= p_dist / c_dist.clamp(min=1e-7)

        idx = torch.where(iou <= iou_thr)[0]
        if idx.numel() == 0:
            break
        order = order[idx]

    return torch.LongTensor(keep)
