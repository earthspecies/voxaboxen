"""
Functions for non-maximal suppression
"""

from typing import Tuple

import numpy as np


def soft_nms(
    bbox_preds: np.ndarray,
    bbox_scores: np.ndarray,
    class_idxs: np.ndarray,
    class_probs: np.ndarray,
    sigma: float = 0.5,
    thresh: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Modified from https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py

    Reference
    https://arxiv.org/abs/1704.04503

    Build a Soft NMS algorithm.
    Arguments
    -------
        bbox_preds: numpy array
            shape=(num_bboxes, 2)

        bbox_scores: numpy array
            shape=(num_bboxes,)

        class_idxs: numpy array
            shape=(num_bboxes,)

        class_probs: numpy array
            shape=(num_bboxes,)

        sigma:       variance of Gaussian function

        thresh:      score thresh

    Returns
    -------
        the index of the selected boxes
    """

    bbox_preds0 = bbox_preds
    bbox_scores0 = bbox_scores
    bbox_preds = bbox_preds.copy()
    bbox_scores = bbox_scores.copy()

    # Indexes concatenate boxes with the last column
    N = bbox_preds.shape[0]

    assert bbox_scores.shape[0] == N
    assert class_idxs.shape[0] == N
    assert class_probs.shape[0] == N

    if N == 0:
        return bbox_preds, bbox_scores, class_idxs, class_probs

    bbox_preds = np.concatenate((bbox_preds, np.arange(0, N)[:, None]), axis=1)

    # The order of boxes coordinate is [start, end]
    start = bbox_preds[:, 0]
    end = bbox_preds[:, 1]

    scores = bbox_scores

    areas = end - start  # compute durations

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].copy()
        pos = i + 1

        if i != N - 1:
            rel_maxpos = np.argmax(scores[pos:], axis=0)
            maxpos = pos + rel_maxpos
            maxscore = scores[maxpos]

            if tscore < maxscore:
                bbox_preds[i], bbox_preds[maxpos] = (
                    bbox_preds[maxpos].copy(),
                    bbox_preds[i].copy(),
                )
                scores[i], scores[maxpos] = scores[maxpos].copy(), scores[i].copy()
                areas[i], areas[maxpos] = areas[maxpos].copy(), areas[i].copy()

        # IoU calculate
        xx1 = np.maximum(bbox_preds[i, 0], bbox_preds[pos:, 0])
        xx2 = np.minimum(bbox_preds[i, 1], bbox_preds[pos:, 1])

        inter = np.maximum(0.0, xx2 - xx1)

        ovr = np.divide(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = np.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep_indices = bbox_preds[:, 2][scores > thresh].astype(int)

    new_bbox_preds = bbox_preds0[keep_indices, :2]
    new_bbox_scores = bbox_scores0[keep_indices]
    new_class_idxs = class_idxs[keep_indices]
    new_class_probs = class_probs[keep_indices]

    return new_bbox_preds, new_bbox_scores, new_class_idxs, new_class_probs


def nms(
    bbox_preds: np.ndarray,
    bbox_scores: np.ndarray,
    class_idxs: np.ndarray,
    class_probs: np.ndarray,
    iou_thresh: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Modified from https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py

    Reference
    https://arxiv.org/abs/1704.04503

    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Parameters
    ----------
    bbox_preds : np.ndarray
        Bounding boxes, shape (num_bboxes, 2)
    bbox_scores : np.ndarray
        Scores for each bounding box, shape (num_bboxes,)
    class_idxs : np.ndarray
        Predicted class indices for each box, shape (num_bboxes,)
    class_probs : np.ndarray
        Predicted class probabilities, shape (num_bboxes,)
    iou_thresh : float, optional
        IoU threshold for suppression, by default 0.5

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Filtered (bbox_preds, bbox_scores, class_idxs, class_probs)
    """

    bbox_preds0 = bbox_preds
    bbox_scores0 = bbox_scores
    bbox_preds = bbox_preds.copy()
    bbox_scores = bbox_scores.copy()

    # Indexes concatenate boxes with the last column
    N = bbox_preds.shape[0]

    assert bbox_scores.shape[0] == N
    assert class_idxs.shape[0] == N
    assert class_probs.shape[0] == N

    if N == 0:
        return bbox_preds, bbox_scores, class_idxs, class_probs

    bbox_preds = np.concatenate((bbox_preds, np.arange(0, N)[:, None]), axis=1)

    # The order of boxes coordinate is [start, end]
    start = bbox_preds[:, 0]
    end = bbox_preds[:, 1]

    scores = bbox_scores

    areas = end - start

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].copy()
        pos = i + 1

        if i != N - 1:
            rel_maxpos = np.argmax(scores[pos:], axis=0)
            maxpos = pos + rel_maxpos
            maxscore = scores[maxpos]

            if tscore < maxscore:
                bbox_preds[i], bbox_preds[maxpos] = (
                    bbox_preds[maxpos].copy(),
                    bbox_preds[i].copy(),
                )
                scores[i], scores[maxpos] = scores[maxpos].copy(), scores[i].copy()
                areas[i], areas[maxpos] = areas[maxpos].copy(), areas[i].copy()

        # IoU calculate
        xx1 = np.maximum(bbox_preds[i, 0], bbox_preds[pos:, 0])
        xx2 = np.minimum(bbox_preds[i, 1], bbox_preds[pos:, 1])

        inter = np.maximum(0.0, xx2 - xx1)

        ovr = np.divide(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = ovr <= iou_thresh
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep_indices = bbox_preds[:, 2][scores > 0].astype(int)
    new_bbox_preds = bbox_preds0[keep_indices, :2]
    new_bbox_scores = bbox_scores0[keep_indices]
    new_class_idxs = class_idxs[keep_indices]
    new_class_probs = class_probs[keep_indices]

    return new_bbox_preds, new_bbox_scores, new_class_idxs, new_class_probs
