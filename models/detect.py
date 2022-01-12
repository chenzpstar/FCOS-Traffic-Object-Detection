# -*- coding: utf-8 -*-
"""
# @file name  : detect.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-11
# @brief      : FCOS检测后处理类
"""

import torch
import torch.nn as nn

from config import FCOSConfig


class FCOSDetect(nn.Module):
    def __init__(self, cfg=None):
        super(FCOSDetect, self).__init__()
        if cfg is None:
            self.cfg = FCOSConfig
        else:
            self.cfg = cfg

        self.use_ctr = self.cfg.use_ctr
        self.strides = self.cfg.strides
        self.score_thr = self.cfg.score_thr
        self.nms_iou_thr = self.cfg.nms_iou_thr
        self.max_boxes_num = self.cfg.max_boxes_num

    def forward(self, preds):
        cls_logits, reg_preds, ctr_logits = preds

        cls_logits = self.reshape_feats(cls_logits)  # bchw -> b(hw)c
        ctr_logits = self.reshape_feats(ctr_logits)  # bchw -> b(hw)c
        cls_preds = cls_logits.sigmoid()
        ctr_preds = ctr_logits.sigmoid()

        cls_scores, cls_ids = cls_preds.max(dim=-1)  # b(hw)c -> b(hw)
        if self.use_ctr:
            cls_scores = (cls_scores * ctr_preds.squeeze(dim=-1)).sqrt()
        cls_ids += 1

        boxes = self._decode_boxes(reg_preds, self.strides)  # bchw -> b(hw)c

        return self._post_process((cls_scores, cls_ids, boxes))

    @staticmethod
    def reshape_feats(feats):
        batch_size, chn_num = feats[0].shape[:2]  # bchw

        feats_reshape = []
        for feat in feats:
            feat = feat.permute(0, 2, 3, 1)  # bchw -> bhwc
            feat = feat.reshape((batch_size, -1, chn_num))  # bhwc -> b(hw)c
            feats_reshape.append(feat)

        return torch.cat(feats_reshape, dim=1)

    def _decode_boxes(self, feats, strides):
        batch_size, chn_num = feats[0].shape[:2]  # bchw

        boxes = []
        for feat, stride in zip(feats, strides):
            coord = self.decode_coords(feat, stride).to(device=feat.device)
            feat = feat.permute(0, 2, 3, 1)  # bchw -> bhwc
            feat = feat.reshape((batch_size, -1, chn_num))  # bhwc -> b(hw)c
            box = self.coords2boxes(coord, feat, stride)
            boxes.append(box)

        return torch.cat(boxes, dim=1)

    @staticmethod
    def decode_coords(feat, stride):
        h, w = feat.shape[2:]  # bchw
        x_shifts = torch.arange(0, w * stride, stride, dtype=float)
        y_shifts = torch.arange(0, h * stride, stride, dtype=float)

        y_shift, x_shift = torch.meshgrid(y_shifts, x_shifts)
        x_shift = x_shift.reshape(-1)
        y_shift = y_shift.reshape(-1)

        return torch.stack([x_shift, y_shift], dim=-1) + stride // 2

    @staticmethod
    def coords2boxes(coords, offsets, stride):
        coords = coords.unsqueeze(dim=0)  # (hw)c -> b(hw)c
        assert len(coords.shape) == len(offsets.shape)

        offsets = offsets * stride
        xy1 = coords - offsets[..., :2]
        xy2 = coords + offsets[..., 2:]

        return torch.cat([xy1, xy2], dim=-1)

    def _post_process(self, preds):
        cls_scores, cls_ids, boxes = preds
        batch_size = cls_scores.shape[0]  # b(hw)

        max_num = min(self.max_boxes_num, cls_scores.shape[-1])
        topk_idx = torch.topk(cls_scores, max_num, dim=-1)[1]
        assert topk_idx.shape == (batch_size, max_num)

        nms_cls_scores = []
        nms_cls_ids = []
        nms_boxes = []

        for i in range(batch_size):
            # 1.挑选topk
            topk_cls_scores = cls_scores[i][topk_idx[i]]
            topk_cls_ids = cls_ids[i][topk_idx[i]]
            topk_boxes = boxes[i][topk_idx[i]]

            # 2.过滤低分
            score_mask = topk_cls_scores > self.score_thr
            filter_cls_scores = topk_cls_scores[score_mask]
            filter_cls_ids = topk_cls_ids[score_mask]
            filter_boxes = topk_boxes[score_mask]

            # 3.计算nms
            nms_idx = self._batch_nms(
                filter_cls_scores,
                filter_cls_ids,
                filter_boxes,
                self.nms_iou_thr,
            )
            nms_cls_scores.append(filter_cls_scores[nms_idx])
            nms_cls_ids.append(filter_cls_ids[nms_idx])
            nms_boxes.append(filter_boxes[nms_idx])

        return nms_cls_scores, nms_cls_ids, nms_boxes

    def _batch_nms(self, cls_scores, cls_ids, boxes, thr):
        if boxes.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        assert boxes.shape[-1] == 4

        coord_max = boxes.max()
        offsets = cls_ids.to(boxes.device) * (coord_max + 1)
        nms_boxes = boxes + offsets.unsqueeze(dim=-1)

        return self.box_nms(cls_scores, nms_boxes, thr)

    @staticmethod
    def box_nms(cls_scores, boxes, thr=0.6):
        if boxes.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        assert boxes.shape[-1] == 4

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
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
            x_min = torch.max(x1[i], x1[order])
            y_min = torch.max(y1[i], y1[order])
            x_max = torch.min(x2[i], x2[order])
            y_max = torch.min(y2[i], y2[order])
            w = (x_max - x_min).clamp(min=0)
            h = (y_max - y_min).clamp(min=0)

            overlap = w * h
            union = areas[i] + areas[order] - overlap
            iou = overlap / union.clamp(min=1e-10)

            idx = (iou <= thr).nonzero().squeeze(dim=-1)
            if idx.numel() == 0:
                break
            order = order[idx]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    model = FCOSDetect()

    preds = [
        [torch.rand(2, 3, 2, 2)] * 5,
        [torch.rand(2, 4, 2, 2)] * 5,
        [torch.rand(2, 1, 2, 2)] * 5,
    ]
    out = model(preds)
    [print(batch_out.shape) for result_out in out for batch_out in result_out]
