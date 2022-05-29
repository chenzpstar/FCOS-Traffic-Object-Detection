# -*- coding: utf-8 -*-
"""
# @file name  : eval.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-17
# @brief      : 评估函数
"""

import numpy as np
import torch
from tqdm import tqdm


def eval_model(data_loader, model, num_classes, device="cpu"):
    # 标签数据的容器
    gt_labels = []
    gt_boxes = []
    # 预测数据的容器
    pred_scores = []
    pred_labels = []
    pred_boxes = []

    # 往两类容器中填值
    for imgs, labels, boxes in tqdm(data_loader):
        with torch.no_grad():
            out = model(imgs.to(device))

        pred_scores.append(out[0][0].cpu().numpy())
        pred_labels.append(out[1][0].cpu().numpy())
        pred_boxes.append(out[2][0].cpu().numpy())
        gt_labels.append(labels[0].numpy())
        gt_boxes.append(boxes[0].numpy())

    # 排序数据
    pred_scores, pred_labels, pred_boxes = sort_by_score(
        pred_scores, pred_labels, pred_boxes)

    # 评估指标
    recalls, precisions, f1s, aps = eval_metrics(
        pred_scores,
        pred_labels,
        pred_boxes,
        gt_labels,
        gt_boxes,
        num_classes,
        0.5,
    )

    return recalls, precisions, f1s, aps


def sort_by_score(pred_scores, pred_labels, pred_boxes):
    score_seq = [(-score).argsort() for score in pred_scores]

    pred_scores = [
        sample_boxes[mask]
        for sample_boxes, mask in zip(pred_scores, score_seq)
    ]
    pred_labels = [
        sample_boxes[mask]
        for sample_boxes, mask in zip(pred_labels, score_seq)
    ]
    pred_boxes = [
        sample_boxes[mask]
        for sample_boxes, mask in zip(pred_boxes, score_seq)
    ]

    return pred_scores, pred_labels, pred_boxes


def _compute_iou(boxes_a, boxes_b):
    """
    numpy 计算IoU
    :param boxes_a: [N,(x1,y1,x2,y2)]
    :param boxes_b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """
    # expands dim
    boxes_a = np.expand_dims(boxes_a, axis=1)  # [N,1,4]
    boxes_b = np.expand_dims(boxes_b, axis=0)  # [1,M,4]
    overlap = np.maximum(
        0.0,
        np.minimum(boxes_a[..., 2:], boxes_b[..., 2:]) -
        np.maximum(boxes_a[..., :2], boxes_b[..., :2]))  # [N,M,(w,h)]

    # overlap
    overlap = np.prod(overlap, axis=-1)  # [N,M]

    # compute area
    area_a = np.prod(boxes_a[..., 2:] - boxes_a[..., :2], axis=-1)
    area_b = np.prod(boxes_b[..., 2:] - boxes_b[..., :2], axis=-1)

    # compute iou
    union = area_a + area_b - overlap
    iou = overlap / np.maximum(union, np.finfo(np.float32).eps)

    return iou


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def eval_metrics(pred_scores,
                 pred_labels,
                 pred_boxes,
                 gt_labels,
                 gt_boxes,
                 num_classes,
                 iou_thr=0.5):
    """
    :param gt_boxes: list of 2d array,shape[(a,(x1,y1,x2,y2)),(b,(x1,y1,x2,y2))...]
    :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
    :param pred_boxes: list of 2d array, shape[(m,(x1,y1,x2,y2)),(n,(x1,y1,x2,y2))...]
    :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
    :param pred_scores: list of 1d array,shape[(m),(n)...]
    :param iou_thr: eg. 0.5
    :param num_classes: eg. 4, total number of class including background which is equal to 0
    :return: a dict containing average precision for each cls
    """
    recalls, precisions, f1s, aps = [], [], [], []
    for label in range(1, num_classes):
        # get samples with specific label
        pred_label_loc = [
            sample_labels == label for sample_labels in pred_labels
        ]
        pred_boxes_cls = [
            sample_boxes[mask]
            for sample_boxes, mask in zip(pred_boxes, pred_label_loc)
        ]
        pred_scores_cls = [
            sample_pred_scores[mask]
            for sample_pred_scores, mask in zip(pred_scores, pred_label_loc)
        ]

        gt_label_loc = [sample_labels == label for sample_labels in gt_labels]
        gt_boxes_cls = [
            sample_boxes[mask]
            for sample_boxes, mask in zip(gt_boxes, gt_label_loc)
        ]

        fp = np.zeros((0, ))
        tp = np.zeros((0, ))
        scores = np.zeros((0, ))
        total_gts = 0

        # loop for each sample
        for sample_pred_scores, sample_pred_boxes, sample_gt_boxes in zip(
                pred_scores_cls, pred_boxes_cls, gt_boxes_cls):
            total_gts = total_gts + len(sample_gt_boxes)
            assigned_gt = []

            # loop for each predicted bbox
            for sample_pred_score, sample_pred_box in zip(
                    sample_pred_scores, sample_pred_boxes):
                scores = np.append(scores, sample_pred_score)

                # if no gts found for the predicted bbox, assign the bbox to fp
                if len(sample_gt_boxes) == 0:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
                    continue

                pred_box = np.expand_dims(sample_pred_box, axis=0)
                iou = _compute_iou(sample_gt_boxes, pred_box)
                gt_for_box = np.argmax(iou, axis=0)
                max_overlap = iou[gt_for_box, 0]

                # one gt can only be assigned to one predicted bbox
                if max_overlap >= iou_thr and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)

        # sort by score
        indices = np.argsort(-scores)
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # compute recall and precision
        recall = tp / np.maximum(total_gts, np.finfo(np.float32).eps)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)
        f1 = 2 * recall[-1] * precision[-1] / np.maximum(
            recall[-1] + precision[-1],
            np.finfo(np.float32).eps)
        ap = _compute_ap(recall, precision)

        recalls.append(recall[-1])
        precisions.append(precision[-1])
        f1s.append(f1)
        aps.append(ap)

    return recalls, precisions, f1s, aps
