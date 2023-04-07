import numpy as np
import csv
import torch
import os
import tqdm
from raven_utils import Clip

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
    new_bbox_preds = bbox_preds0[keep_indices, :2]

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
  # model.eval()
  
  all_predictions = []
  with torch.no_grad():
    for i, (X, _, _) in tqdm.tqdm(enumerate(dataloader)):
      X = torch.Tensor(X).to(device = device, dtype = torch.float)
      # print(X)
      # print(model(X)[0,:,0])
      predictions = torch.sigmoid(model(X)) #[batch, time, channels]
      # print(predictions[0,:,0])
      all_predictions.append(predictions)
  all_predictions = torch.cat(all_predictions)
  all_predictions = torch.reshape(all_predictions, (-1, all_predictions.size(-1)))
  return all_predictions.detach().cpu().numpy()

def export_to_selection_table(predictions, fn, args):
  anchor_preds = predictions > args.detection_threshold
  print(f"found {np.sum(anchor_preds)} possible boxes")
  anchor_scores = predictions
  anchor_win_sizes = args.anchor_durs_sec
  pred_sr = args.sr // (args.scale_factor * args.prediction_scale_factor)
  
  bboxes, scores = pred2bbox(anchor_preds, anchor_scores, anchor_win_sizes, pred_sr)
  snms_bboxes, _ = soft_nms(bboxes, scores, sigma=0.5, thresh=0.001)
  nms_bboxes, _ = nms(bboxes, scores, iou_thresh=0.5)
  
  target_fp = os.path.join(args.experiment_dir, f"snms_pred_{fn}.txt")
  st = bbox2raven(snms_bboxes, "crow")
  write_tsv(target_fp, st)  
  
  target_fp = os.path.join(args.experiment_dir, f"nms_pred_{fn}.txt")
  st = bbox2raven(nms_bboxes, "crow")
  write_tsv(target_fp, st)
  return target_fp
  
def get_metrics(predictions_fp, annotations_fp):
  c = Clip()
  
  # Hard coded for now
  label_mapping = {
        'focal': 'crow',
        'focal?': 'crow',
        'not focal': 'crow',
        'not focal LD': 'crow',
        'not focal?': 'crow',
        'crowchicks': 'crow',
        'crow_undeter': 'crow',
        'nest': 'crow',
    }
  
  c.load_predictions(predictions_fp)
  c.load_annotations(annotations_fp, label_mapping = label_mapping)
  
  metrics = {}
  
  for iou_thresh in [0.2, 0.5, 0.8]:
    c.compute_matching(IoU_minimum = iou_thresh)
    metrics[iou_thresh] = c.evaluate()
  
  return metrics

def summarize_metrics(metrics):
  # metrics (dict) : {fp : fp_metrics}
  # where
  # metrics_dict (dict) : {iou_thresh : {'TP': int, 'FP' : int, 'FN' : int}}
  # import pdb; pdb.set_trace()
  
  fps = sorted(metrics.keys())
  iou_thresholds = sorted(metrics[fps[0]].keys())
  
  overall = {iou_thresh : {'TP' : 0, 'FP' : 0, 'FN' : 0} for iou_thresh in iou_thresholds}
  
  for fp in fps:
    for iou_thresh in iou_thresholds:
      counts = metrics[fp][iou_thresh]
      overall[iou_thresh]['TP'] += counts['TP']
      overall[iou_thresh]['FP'] += counts['FP']
      overall[iou_thresh]['FN'] += counts['FN']
      
  for iou_thresh in iou_thresholds:
    tp = overall[iou_thresh]['TP']
    fp = overall[iou_thresh]['FP']
    fn = overall[iou_thresh]['FN']
    
    if tp + fp == 0:
      prec = 1
    else:
      prec = tp / (tp + fp)
    overall[iou_thresh]['precision'] = prec
    
    if tp + fn == 0:
      rec = 1
    else:
      rec = tp / (tp + fn)
    overall[iou_thresh]['recall'] = rec
      
    if prec + rec == 0:
      f1 = 0
    else:
      f1 = 2*prec*rec / (prec + rec)
    overall[iou_thresh]['f1'] = f1
  
  return overall

def predict_and_evaluate(model, dataloader_dict, args):
  metrics = {}
  for fn in dataloader_dict:
    predictions = generate_predictions(model, dataloader_dict[fn], args)
    predictions_fp = export_to_selection_table(predictions, fn, args)
    annotations_fp = os.path.join(args.annotation_selection_tables_dir, f"{fn}.txt")
    metrics[fn] = get_metrics(predictions_fp, annotations_fp)
  
  summary = summarize_metrics(metrics)
  metrics['summary'] = summary
  return metrics
