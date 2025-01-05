import numpy as np

def soft_nms(bbox_preds, bbox_scores, sigma=0.5, thresh=0.001):
    """
    Modified from https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py

    Reference
    https://arxiv.org/abs/1704.04503

    Build a Soft NMS algorithm.
    # Augments
        bbox_preds: numpy array
            shape=(num_bboxes, 2)

        bbox_scores: numpy array
            shape=(num_bboxes,)

        sigma:       variance of Gaussian function

        thresh:      score thresh

    # Return
        the index of the selected boxes
    """

    bbox_preds0 = bbox_preds
    bbox_preds = bbox_preds.copy()
    bbox_scores = bbox_scores.copy()

    # Indexes concatenate boxes with the last column
    N = bbox_preds.shape[0]

    assert (bbox_scores.shape[0] == N)

    if N == 0:
        return None, []

    bbox_preds = np.concatenate((bbox_preds, np.arange(0, N)[:, None]), axis=1)

    # The order of boxes coordinate is [start, end]
    start = bbox_preds[:, 0]
    end = bbox_preds[:, 1]

    scores = bbox_scores

    areas = end-start

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].copy()
        pos = i + 1

        if i != N - 1:
            rel_maxpos = np.argmax(scores[pos:], axis=0)
            maxpos = pos + rel_maxpos
            maxscore = scores[maxpos]

            if tscore < maxscore:
                bbox_preds[i], bbox_preds[maxpos] = bbox_preds[maxpos].copy(), bbox_preds[i].copy()
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

    return new_bbox_preds, keep_indices


def nms(bbox_preds, bbox_scores, iou_thresh=0.5):
    """
    Modified from https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py

    Reference
    https://arxiv.org/abs/1704.04503

    Build a Soft NMS algorithm.
    # Augments
        bbox_preds: numpy array
            shape=(num_bboxes, 2)

        bbox_scores: numpy array
            shape=(num_bboxes,)

        thresh:      score thresh

    # Return
        the index of the selected boxes
    """

    bbox_preds0 = bbox_preds
    bbox_preds = bbox_preds.copy()
    bbox_scores = bbox_scores.copy()

    # Indexes concatenate boxes with the last column
    N = bbox_preds.shape[0]

    assert (bbox_scores.shape[0] == N)

    if N == 0:
        return None, []

    bbox_preds = np.concatenate((bbox_preds, np.arange(0, N)[:, None]), axis=1)

    # The order of boxes coordinate is [start, end]
    start = bbox_preds[:, 0]
    end = bbox_preds[:, 1]

    scores = bbox_scores

    areas = end-start

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].copy()
        pos = i + 1

        if i != N - 1:
            rel_maxpos = np.argmax(scores[pos:], axis=0)
            maxpos = pos + rel_maxpos
            maxscore = scores[maxpos]

            if tscore < maxscore:
                bbox_preds[i], bbox_preds[maxpos] = bbox_preds[maxpos].copy(), bbox_preds[i].copy()
                scores[i], scores[maxpos] = scores[maxpos].copy(), scores[i].copy()
                areas[i], areas[maxpos] = areas[maxpos].copy(), areas[i].copy()

        # IoU calculate
        xx1 = np.maximum(bbox_preds[i, 0], bbox_preds[pos:, 0])
        xx2 = np.minimum(bbox_preds[i, 1], bbox_preds[pos:, 1])

        inter = np.maximum(0.0, xx2 - xx1)

        ovr = np.divide(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = (ovr < iou_thresh)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep_indices = bbox_preds[:, 2][scores > 0].astype(int)
    new_bbox_preds = bbox_preds0[keep_indices, :2]

    return new_bbox_preds, keep_indices
