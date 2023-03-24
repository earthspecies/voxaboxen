import numpy as np
import csv
import torch
import os
import tqdm
# import pdb
device = "cuda" if torch.cuda.is_available() else "cpu"

def pred2bbox(anchor_preds, anchor_scores, anchor_win_sizes, pred_sr):
    '''
    anchor_preds:
        shape=(num_frames, num_anchors)

    anchor_scores:
        shape=(num_frames, num_anchors)

    anchor_win_sizes: list
        anchor window sizes in seconds
        len(anchor_win_sizes) == num_anchors

    pred_sr:
        prediction sampling rate in Hz

    '''

    bboxes = []
    scores = []
    for center_idx, pred in enumerate(anchor_preds):
        for anchor_idx in pred.nonzero()[0]:
            win_size = anchor_win_sizes[anchor_idx]

            half_win_size = win_size / 2.

            center = center_idx / pred_sr

            bbox = [max(0, center-half_win_size), center+half_win_size]

            score = anchor_scores[center_idx, anchor_idx]

            bboxes.append(bbox)
            scores.append(score)

    bboxes = np.array(bboxes)
    scores = np.array(scores)

    return bboxes, scores


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
            maxpos = np.argmax(scores[pos:], axis=0)
            maxscore = scores[maxpos]

            if tscore < maxscore:
                bbox_preds[i], bbox_preds[maxpos + i + 1] = bbox_preds[maxpos + i + 1].copy(), bbox_preds[i].copy()
                scores[i], scores[maxpos + i + 1] = scores[maxpos + i + 1].copy(), scores[i].copy()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].copy(), areas[i].copy()

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
    new_bbox_preds = bbox_preds[keep_indices, :2]

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

        sigma:       variance of Gaussian function

        thresh:      score thresh

    # Return
        the index of the selected boxes
    """

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
            maxpos = np.argmax(scores[pos:], axis=0)
            maxscore = scores[maxpos]

            if tscore < maxscore:
                bbox_preds[i], bbox_preds[maxpos + i + 1] = bbox_preds[maxpos + i + 1].copy(), bbox_preds[i].copy()
                scores[i], scores[maxpos + i + 1] = scores[maxpos + i + 1].copy(), scores[i].copy()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].copy(), areas[i].copy()

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
    new_bbox_preds = bbox_preds[keep_indices, :2]

    return new_bbox_preds, keep_indices


def bbox2raven(bboxes, labels=None):
    '''
    output bounding boxes to a selection table

    out_fp:
        output file path

    bboxes: numpy array
        shape=(num_bboxes, 2)

    labels: str or list

    '''
    if bboxes is None:
      return [['Begin Time (s)', 'End Time (s)', 'Annotation']]
    
    N = bboxes.shape[0]

    if labels is None:
        labels = ['' for ii in range(N)]
    elif type(labels) == list:
        assert (len(labels) == N)
    elif type(labels) == str:
        labels = [labels for ii in range(N)]

    columns = ['Begin Time (s)', 'End Time (s)', 'Annotation']

    out_data = [[bbox[0], bbox[1], label] for bbox, label in zip(bboxes, labels)]
    out_data = sorted(out_data, key=lambda x: x[:2])

    out = [columns] + out_data

    return out

def write_tsv(out_fp, data):
    '''
    out_fp:
        output file path

    data: list of lists
    '''

    with open(out_fp, 'w', newline='') as ff:
        tsv_output = csv.writer(ff, delimiter='\t')

        for row in data:
            tsv_output.writerow(row)

  
def generate_predictions(model, dataloader, args):
  model = model.to(device)
  model.eval()
  
  all_predictions = []
  with torch.no_grad():
    for i, (X, _, _) in tqdm.tqdm(enumerate(dataloader)):
      X = torch.Tensor(X).to(device = device, dtype = torch.float)
      predictions = torch.sigmoid(model(X)) #[batch, time, channels]
      all_predictions.append(predictions)
  all_predictions = torch.cat(all_predictions)
  all_predictions = torch.reshape(all_predictions, (-1, all_predictions.size(-1)))
  return all_predictions.detach().cpu().numpy()

def export_to_selection_table(predictions, fn, args):
  anchor_preds = predictions > 0.5
  print(f"found {np.sum(anchor_preds)} possible boxes")
  anchor_scores = predictions
  anchor_win_sizes = args.anchor_durs_sec
  pred_sr = args.sr // args.scale_factor
  bboxes, scores = pred2bbox(anchor_preds, anchor_scores, anchor_win_sizes, pred_sr)
  snms_bboxes, _ = soft_nms(bboxes, scores, sigma=0.5, thresh=0.001)
  nms_bboxes, _ = nms(bboxes, scores, iou_thresh=0.5)
  
  target_fp = os.path.join(args.experiment_dir, f"snms_pred_{fn}.txt")
  st = bbox2raven(snms_bboxes, "crow")
  write_tsv(target_fp, st)  
  
  target_fp = os.path.join(args.experiment_dir, f"nms_pred_{fn}.txt")
  st = bbox2raven(nms_bboxes, "crow")
  write_tsv(target_fp, st)  
            
if __name__ == "__main__":
    anchor_preds = np.array(
        [
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    anchor_scores = np.array(
        [
            [0, 0.6, 0.7],
            [0.5, 0.4, 0],
            [0.9, 0, 0],
            [0, 0.8, 0],
        ]
    )

    anchor_win_sizes = [0.1, 0.3, 0.5]

    labels = 'crow'

    pred_sr = 5

    bboxes, scores = pred2bbox(anchor_preds, anchor_scores, anchor_win_sizes, pred_sr)
    print(bboxes)
    print(scores)

    snms_bboxes, _ = soft_nms(bboxes, scores, sigma=0.5, thresh=0.001)
    nms_bboxes, _ = nms(bboxes, scores, iou_thresh=0.5)

    print(snms_bboxes)
    print(nms_bboxes)

    st = bbox2raven(snms_bboxes, labels)
    print(st)
    write_tsv('st.txt', st)
