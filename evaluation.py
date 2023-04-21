import numpy as np
import csv
import torch
import os
import tqdm
from raven_utils import Clip
from model import preprocess_and_augment
from scipy.signal import find_peaks

device = "cuda" if torch.cuda.is_available() else "cpu"

def pred2bbox(anchor_preds, anchor_scores, regressions, pred_sr):
    '''
    anchor_preds:
        shape=(num_frames,)

    anchor_scores:
        shape=(num_frames,)

    regressions:
        shape=(num_frames, 2)

    pred_sr:
        prediction sampling rate in Hz

    '''
    bboxes = []
    scores = []
    for start_idx, pred in enumerate(anchor_preds):   
        if pred:
          duration = regressions[start_idx]
          
          if duration <= 0:
            continue

          start = start_idx / pred_sr
          bbox = [start, start+duration]

          score = anchor_scores[start_idx]

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
  model.eval()
  
  all_predictions = []
  all_regressions = []
  with torch.no_grad():
    for i, (X, _, _, _) in tqdm.tqdm(enumerate(dataloader)):
      X = torch.Tensor(X).to(device = device, dtype = torch.float)
      X, _, _ = preprocess_and_augment(X, None, None, False, args)
      predictions, regression = model(X)
      # predictions = torch.sigmoid(logits)
      all_predictions.append(predictions)
      all_regressions.append(regression)
    all_predictions = torch.cat(all_predictions)
    all_regressions = torch.cat(all_regressions)

    # we use half overlapping windows, need to throw away boundary predictions
    
    ######## Need better checking that preds are the correct dur    
    assert all_predictions.size(dim=1) % 2 == 0
    first_quarter_window_dur_samples=all_predictions.size(dim=1)//4
    last_quarter_window_dur_samples=(all_predictions.size(dim=1)//2)-first_quarter_window_dur_samples
    
    # assemble predictions
    beginning_bit = all_predictions[0,:first_quarter_window_dur_samples]
    end_bit = all_predictions[-1,-last_quarter_window_dur_samples:]
    predictions_clipped = all_predictions[:,first_quarter_window_dur_samples:-last_quarter_window_dur_samples]
    all_predictions = torch.flatten(predictions_clipped)
    all_predictions = torch.cat([beginning_bit, all_predictions, end_bit])
    
    # assemble regressions
    beginning_bit = all_regressions[0,:first_quarter_window_dur_samples]
    end_bit = all_regressions[-1,-last_quarter_window_dur_samples:]
    regressions_clipped = all_regressions[:,first_quarter_window_dur_samples:-last_quarter_window_dur_samples]
    all_regressions = torch.flatten(regressions_clipped)
    all_regressions = torch.cat([beginning_bit, all_regressions, end_bit])
    
  return all_predictions.detach().cpu().numpy(), all_regressions.detach().cpu().numpy()

def export_to_selection_table(predictions, regressions, fn, args):
  target_fp = os.path.join(args.experiment_dir, f"probs_{fn}.npy")
  np.save(target_fp, predictions)
  
  target_fp = os.path.join(args.experiment_dir, f"regressions_{fn}.npy")
  np.save(target_fp, regressions)
  
  # # nms
  # anchor_preds = predictions > args.detection_threshold
  # print(f"NMS: found {np.sum(anchor_preds)} possible boxes")
  # anchor_scores = predictions
  # pred_sr = args.sr // (args.scale_factor * args.prediction_scale_factor)
  # bboxes, scores = pred2bbox(anchor_preds, anchor_scores, regressions, pred_sr)
  # snms_bboxes, _ = soft_nms(bboxes, scores, sigma=0.5, thresh=0.001)
  # nms_bboxes, _ = nms(bboxes, scores, iou_thresh=0.5)
  # print(f"SNMS found {len(snms_bboxes)} boxes")
  # print(f"NMS found {len(nms_bboxes)} boxes")
  
  # peaks
  peaks, _ = find_peaks(predictions, height=args.detection_threshold, distance=5)
  pred_sr = args.sr // (args.scale_factor * args.prediction_scale_factor)
  print(f"Peaks found {len(peaks)} boxes")
  anchor_peak_preds = np.zeros(predictions.shape, dtype='bool')
  anchor_peak_preds[peaks] = True
  anchor_scores = predictions
  peaks_bboxes, scores = pred2bbox(anchor_peak_preds, anchor_scores, regressions, pred_sr)
  
#   target_fp = os.path.join(args.experiment_dir, f"snms_pred_{fn}.txt")
#   st = bbox2raven(snms_bboxes, "crow")
#   write_tsv(target_fp, st)  
  
#   target_fp = os.path.join(args.experiment_dir, f"nms_pred_{fn}.txt")
#   st = bbox2raven(nms_bboxes, "crow")
#   write_tsv(target_fp, st)
  
  target_fp = os.path.join(args.experiment_dir, f"peaks_pred_{fn}.txt")
  st = bbox2raven(peaks_bboxes, "crow")
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
    predictions, regressions = generate_predictions(model, dataloader_dict[fn], args)
    predictions_fp = export_to_selection_table(predictions, regressions, fn, args)
    annotations_fp = os.path.join(args.annotation_selection_tables_dir, f"{fn}.txt")
    metrics[fn] = get_metrics(predictions_fp, annotations_fp)
  
  summary = summarize_metrics(metrics)
  metrics['summary'] = summary
  return metrics
