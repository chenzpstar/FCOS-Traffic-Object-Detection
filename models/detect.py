# -*- coding: utf-8 -*-
"""
# @file name  : detect.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-11
# @brief      : FCOS检测后处理类
"""

import torch
import torch.nn as nn

try:
    from .config import FCOSConfig
    from .utils import clip_boxes, decode_preds, nms_boxes, reshape_feats
except:
    from config import FCOSConfig
    from utils import clip_boxes, decode_preds, nms_boxes, reshape_feats


class FCOSDetect(nn.Module):
    def __init__(self, cfg=None):
        super(FCOSDetect, self).__init__()
        self.cfg = FCOSConfig if cfg is None else cfg
        self.use_ctr = self.cfg.use_ctr
        self.strides = self.cfg.strides
        self.max_boxes_num = self.cfg.max_boxes_num
        self.score_thr = self.cfg.score_thr
        self.nms_iou_thr = self.cfg.nms_iou_thr
        self.nms_mode = self.cfg.nms_mode

    def forward(self, preds, imgs):
        cls_logits, reg_preds, ctr_logits = preds

        cls_logits = reshape_feats(cls_logits)  # bchw -> b(hw)c
        ctr_logits = reshape_feats(ctr_logits)  # bchw -> b(hw)c
        cls_preds = torch.sigmoid(cls_logits)
        ctr_preds = torch.sigmoid(ctr_logits)

        cls_scores, pred_labels = cls_preds.max(dim=-1)  # b(hw)c -> b(hw)
        if self.use_ctr:
            cls_scores = torch.sqrt(cls_scores * ctr_preds.squeeze_(dim=-1))
        pred_labels += 1

        pred_boxes = decode_preds(reg_preds, self.strides)  # bchw -> b(hw)c

        return self._post_process((cls_scores, pred_labels, pred_boxes), imgs)

    def _post_process(self, preds, imgs):
        cls_scores, pred_labels, pred_boxes = preds
        batch_size, hw = cls_scores.shape  # b(hw)

        max_boxes_num = min(self.max_boxes_num, hw)
        topk_idx = torch.topk(cls_scores, max_boxes_num, dim=-1)[1]
        assert topk_idx.shape == (batch_size, max_boxes_num)

        nms_cls_scores = []
        nms_pred_labels = []
        nms_pred_boxes = []

        for i in range(batch_size):
            # 1. 挑选topk
            topk_cls_scores = cls_scores[i][topk_idx[i]]
            topk_pred_labels = pred_labels[i][topk_idx[i]]
            topk_pred_boxes = pred_boxes[i][topk_idx[i]]

            # 2. 过滤低分
            score_mask = topk_cls_scores >= self.score_thr
            filter_cls_scores = topk_cls_scores[score_mask]
            filter_pred_labels = topk_pred_labels[score_mask]
            filter_pred_boxes = topk_pred_boxes[score_mask]

            # 3. 计算nms
            nms_idx = self._batch_nms(
                filter_pred_boxes,
                filter_cls_scores,
                filter_pred_labels,
                self.nms_iou_thr,
                self.nms_mode,
            )
            nms_cls_scores.append(filter_cls_scores[nms_idx])
            nms_pred_labels.append(filter_pred_labels[nms_idx])
            nms_pred_boxes.append(filter_pred_boxes[nms_idx])

        nms_pred_boxes = list(map(clip_boxes, nms_pred_boxes, imgs))

        return nms_cls_scores, nms_pred_labels, nms_pred_boxes

    def _batch_nms(self, boxes, scores, labels, iou_thr, mode):
        # Strategy: in order to perform NMS independently per class,
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap.
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        assert boxes.shape[-1] == 4

        max_coord, min_coord = boxes.max(), boxes.min()
        offsets = labels.to(boxes) * (max_coord - min_coord + 1)
        new_boxes = boxes.clone() + offsets[:, None]

        return nms_boxes(new_boxes, scores, iou_thr, mode)


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    model = FCOSDetect()

    imgs = torch.rand(2, 3, 224, 224)
    preds = (
        [torch.rand(2, 3, 2, 2)] * 5,
        [torch.rand(2, 4, 2, 2)] * 5,
        [torch.rand(2, 1, 2, 2)] * 5,
    )

    out = model(imgs, preds)
    [print(batch_out.shape) for result_out in out for batch_out in result_out]
