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

eps = 1e-7


def eval_model(model,
               data_loader,
               num_classes,
               iou_thr=0.5,
               use_07_metric=False,
               device="cpu"):
    pred_scores = []
    pred_labels = []
    pred_boxes = []
    gt_labels = []
    gt_boxes = []

    # 1. 预测数据
    for imgs, labels, boxes in tqdm(data_loader):
        imgs = imgs.to(device, non_blocking=True)

        with torch.no_grad():
            outs = model(imgs)

        pred_scores.extend(map(lambda scores: scores.cpu().numpy(), outs[0]))
        pred_labels.extend(map(lambda labels: labels.cpu().numpy(), outs[1]))
        pred_boxes.extend(map(lambda boxes: boxes.cpu().numpy(), outs[2]))
        gt_labels.extend(labels.numpy())
        gt_boxes.extend(boxes.numpy())

    # 2. 排序数据
    pred_scores, pred_labels, pred_boxes = sort_by_score(
        pred_scores, pred_labels, pred_boxes)

    # 3. 评估指标
    metrics = eval_metrics(pred_boxes, pred_scores, pred_labels, gt_boxes,
                           gt_labels, num_classes, iou_thr, use_07_metric)

    return metrics


def sort_by_score(scores, labels, boxes):
    orders = tuple(map(lambda score: (-score).argsort(), scores))
    sorted_scores = tuple(
        map(lambda score, order: score[order], scores, orders))
    sorted_labels = tuple(
        map(lambda label, order: label[order], labels, orders))
    sorted_boxes = tuple(map(lambda box, order: box[order], boxes, orders))

    return sorted_scores, sorted_labels, sorted_boxes


def _compute_iou(boxes_a, boxes_b):
    """
    numpy 计算IoU
    :param boxes_a: [N,(x1,y1,x2,y2)]
    :param boxes_b: [M,(x1,y1,x2,y2)]
    :return: IoU [N,M]
    """
    # expands dim
    boxes_a = np.expand_dims(boxes_a, axis=1)  # [N,1,4]
    boxes_b = np.expand_dims(boxes_b, axis=0)  # [1,M,4]

    # compute overlap
    xy1 = np.maximum(boxes_a[..., :2], boxes_b[..., :2])  # [N,M,2]
    xy2 = np.minimum(boxes_a[..., 2:], boxes_b[..., 2:])  # [N,M,2]
    wh = (xy2 - xy1).clip(0.0)  # [N,M,2]
    overlap = np.prod(wh, axis=-1)  # [N,M]

    # compute area
    area_a = np.prod(boxes_a[..., 2:] - boxes_a[..., :2], axis=-1)  # [N,M]
    area_b = np.prod(boxes_b[..., 2:] - boxes_b[..., :2], axis=-1)  # [N,M]

    # compute iou
    union = area_a + area_b - overlap  # [N,M]
    iou = overlap / union.clip(eps)  # [N,M]

    return iou


def _compute_ap(recall, precision, use_07_metric=False):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mprec = np.concatenate(([0.0], precision, [0.0]))

        # compute the precision envelope
        for i in range(mprec.size - 1, 0, -1):
            mprec[i - 1] = np.maximum(mprec[i - 1], mprec[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * precision
        ap = np.sum((mrec[i + 1] - mrec[i]) * mprec[i + 1])

    return ap


def eval_metrics(pred_boxes,
                 pred_scores,
                 pred_labels,
                 gt_boxes,
                 gt_labels,
                 num_classes,
                 iou_thr=0.5,
                 use_07_metric=False):
    """
    :param pred_boxes: list of 2d array, shape[(m,(x1,y1,x2,y2)),(n,(x1,y1,x2,y2))...]
    :param pred_scores: list of 1d array, shape[(m),(n)...]
    :param pred_labels: list of 1d array, shape[(m),(n)...], value is sparse label index
    :param gt_boxes: list of 2d array, shape[(a,(x1,y1,x2,y2)),(b,(x1,y1,x2,y2))...]
    :param gt_labels: list of 1d array, shape[(a),(b)...], value is sparse label index
    :param num_classes: eg. 3, total number of classes, excluding background which is equal to 0
    :param iou_thr: eg. 0.5
    :return: a series of metrics for each cls
    """
    metric_rec = {"rec": [], "prec": [], "f1": [], "ap": []}

    for label in range(1, num_classes + 1):
        # get samples with specific label
        pred_label_loc = tuple(map(lambda labels: labels == label,
                                   pred_labels))
        pred_boxes_cls = tuple(
            map(lambda boxes, mask: boxes[mask], pred_boxes, pred_label_loc))
        pred_scores_cls = tuple(
            map(lambda scores, mask: scores[mask], pred_scores,
                pred_label_loc))

        gt_label_loc = tuple(map(lambda labels: labels == label, gt_labels))
        gt_boxes_cls = tuple(
            map(lambda boxes, mask: boxes[mask], gt_boxes, gt_label_loc))

        fp = np.zeros((0, ))
        tp = np.zeros((0, ))
        scores = np.zeros((0, ))
        total_gt_boxes = 0

        # loop for each sample
        for sample_pred_scores, sample_pred_boxes, sample_gt_boxes in zip(
                pred_scores_cls, pred_boxes_cls, gt_boxes_cls):
            total_gt_boxes += len(sample_gt_boxes)
            assigned_gt_boxes = []

            # loop for each predicted bbox
            for sample_pred_score, sample_pred_box in zip(
                    sample_pred_scores, sample_pred_boxes):
                scores = np.append(scores, sample_pred_score)

                # if no gts found for the predicted bbox, assign the bbox to fp
                if len(sample_gt_boxes) == 0:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
                    continue

                sample_pred_box = np.expand_dims(sample_pred_box, axis=0)
                iou = _compute_iou(sample_gt_boxes, sample_pred_box)
                max_iou_idx = np.argmax(iou, axis=0)
                max_iou = iou[max_iou_idx, 0]

                # one gt can only be assigned to one predicted bbox
                if max_iou >= iou_thr and max_iou_idx not in assigned_gt_boxes:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt_boxes.append(max_iou_idx)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)

        # sort by score
        order = np.argsort(-scores)
        fp = fp[order]
        tp = tp[order]
        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # compute recall, precision and f1 score
        rec = tp / np.maximum(total_gt_boxes, eps)
        prec = tp / (tp + fp).clip(eps)
        f1 = 2 * rec[-1] * prec[-1] / (rec[-1] + prec[-1]).clip(eps)
        # compute average precision
        ap = _compute_ap(rec, prec, use_07_metric)

        metric_rec["rec"].append(rec[-1])
        metric_rec["prec"].append(prec[-1])
        metric_rec["f1"].append(f1)
        metric_rec["ap"].append(ap)

    return metric_rec
