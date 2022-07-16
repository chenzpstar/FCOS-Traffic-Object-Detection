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
    from .utils import clip_boxes, decode_boxes, nms_boxes
except:
    from config import FCOSConfig
    from utils import clip_boxes, decode_boxes, nms_boxes


class FCOSDetect(nn.Module):
    def __init__(self, cfg=None):
        super(FCOSDetect, self).__init__()
        self.cfg = FCOSConfig if cfg is None else cfg
        self.strides = self.cfg.strides
        self.use_ctrness = self.cfg.use_ctrness
        self.max_num_boxes = self.cfg.max_num_boxes
        self.score_thr = self.cfg.score_thr
        self.nms_iou_thr = self.cfg.nms_iou_thr
        self.nms_method = self.cfg.nms_method

    def forward(self, preds, imgs):
        cls_logits, reg_preds, ctr_logits, coords = preds

        cls_logits = torch.cat(cls_logits, dim=1)
        ctr_logits = torch.cat(ctr_logits, dim=1)
        cls_preds = torch.sigmoid(cls_logits)
        ctr_preds = torch.sigmoid(ctr_logits)

        pred_scores, pred_labels = cls_preds.max(dim=-1)  # b(hw)c -> b(hw)
        if self.use_ctrness:
            pred_scores = torch.sqrt(pred_scores * ctr_preds.squeeze_(dim=-1))
        pred_labels += 1

        pred_boxes = decode_boxes(reg_preds, coords, self.strides)
        pred_boxes = clip_boxes(pred_boxes, imgs)

        return self._post_process(pred_scores, pred_labels, pred_boxes)

    def _post_process(self, scores, labels, boxes):
        batch_size, num_points = scores.shape  # b(hw)

        max_num_boxes = min(self.max_num_boxes, num_points)
        topk_idx = torch.topk(scores, max_num_boxes, dim=-1)[1]
        assert topk_idx.shape == (batch_size, max_num_boxes)

        nms_scores = []
        nms_labels = []
        nms_boxes = []

        for i in range(batch_size):
            # 1. 挑选topk
            topk_scores = scores[i][topk_idx[i]]
            topk_labels = labels[i][topk_idx[i]]
            topk_boxes = boxes[i][topk_idx[i]]

            # 2. 过滤低分
            score_mask = topk_scores >= self.score_thr
            filter_scores = topk_scores[score_mask]
            filter_labels = topk_labels[score_mask]
            filter_boxes = topk_boxes[score_mask]

            # 3. 计算nms
            nms_idx = self._batch_nms(filter_boxes, filter_scores,
                                      filter_labels, self.nms_iou_thr,
                                      self.nms_method)
            nms_scores.append(filter_scores[nms_idx])
            nms_labels.append(filter_labels[nms_idx])
            nms_boxes.append(filter_boxes[nms_idx])

        return nms_scores, nms_labels, nms_boxes

    def _batch_nms(self, boxes, scores, labels, iou_thr=0.5, method="iou"):
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

        return nms_boxes(new_boxes, scores, iou_thr, method)


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    model = FCOSDetect()

    imgs = torch.rand(2, 3, 224, 224)
    preds = (
        [torch.rand(2, 4, 3)] * 5,
        [torch.rand(2, 4, 4)] * 5,
        [torch.rand(2, 4, 1)] * 5,
        [torch.rand(4, 2)] * 5,
    )

    out = model(preds, imgs)
    [print(batch_out.shape) for result_out in out for batch_out in result_out]
