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
        shape=(num_frames, n_classes)

    anchor_scores:
        shape=(num_frames, n_classes)

    regressions:
        shape=(num_frames, n_classes)

    pred_sr:
        prediction sampling rate in Hz

    '''
    bboxes = []
    scores = []
    class_idxs = []
    
    ### Stopped here
    
    pred_locs = np.nonzero(anchor_preds)
    
    for x in range(len(pred_locs[0])):
        i = pred_locs[0][x]
        j = pred_locs[1][x]
        duration = regressions[i, j]
        if duration <= 0:
          continue
          
        start = i / pred_sr
        bbox = [start, start+duration]
        
        score = anchor_scores[i, j]
        
        class_idx = j
        
        bboxes.append(bbox)
        scores.append(score)
        class_idxs.append(class_idx)
        
    bboxes = np.array(bboxes)
    scores = np.array(scores)
    class_idxs = np.array(class_idxs)
    
    
    return bboxes, scores, class_idxs
    
    
#     for start_idx, pred in enumerate(anchor_preds):   
#         if pred:
#           duration = regressions[start_idx]
          
#           if duration <= 0:
#             continue

#           start = start_idx / pred_sr
#           bbox = [start, start+duration]

#           score = anchor_scores[start_idx]

#           bboxes.append(bbox)
#           scores.append(score)

#     bboxes = np.array(bboxes)
#     scores = np.array(scores)

#     return bboxes, scores



def bbox2raven(bboxes, class_idxs, label_set):
    '''
    output bounding boxes to a selection table

    out_fp:
        output file path

    bboxes: numpy array
        shape=(num_bboxes, 2)
        
    class_idxs: numpy array
        shape=(num_bboxes,)

    label_set: list

    '''
    if bboxes is None:
      return [['Begin Time (s)', 'End Time (s)', 'Annotation']]

    columns = ['Begin Time (s)', 'End Time (s)', 'Annotation']
        
    out_data = [[bbox[0], bbox[1], label_set[int(c)]] for bbox, c in zip(bboxes, class_idxs)]
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
      X, _, _, _ = preprocess_and_augment(X, None, None, None, False, args)
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
    beginning_bit = all_predictions[0,:first_quarter_window_dur_samples,:]
    end_bit = all_predictions[-1,-last_quarter_window_dur_samples:,:]
    predictions_clipped = all_predictions[:,first_quarter_window_dur_samples:-last_quarter_window_dur_samples,:]
    all_predictions = torch.reshape(predictions_clipped, (-1, predictions_clipped.size(-1)))
    all_predictions = torch.cat([beginning_bit, all_predictions, end_bit])
    
    # assemble regressions
    beginning_bit = all_regressions[0,:first_quarter_window_dur_samples, :]
    end_bit = all_regressions[-1,-last_quarter_window_dur_samples:, :]
    regressions_clipped = all_regressions[:,first_quarter_window_dur_samples:-last_quarter_window_dur_samples,:]
    all_regressions = torch.reshape(regressions_clipped, (-1, regressions_clipped.size(-1)))
    all_regressions = torch.cat([beginning_bit, all_regressions, end_bit])
    
  return all_predictions.detach().cpu().numpy(), all_regressions.detach().cpu().numpy()

def export_to_selection_table(predictions, regressions, fn, args):
  target_fp = os.path.join(args.experiment_dir, f"probs_{fn}.npy")
  np.save(target_fp, predictions)
  
  target_fp = os.path.join(args.experiment_dir, f"regressions_{fn}.npy")
  np.save(target_fp, regressions)
  
  ## peaks  
  # uses two masks to enforce we don't have multiple classes for one peak
  all_classes_max = np.amax(predictions, axis = -1)
  all_classes_peaks, _ = find_peaks(all_classes_max, height = args.detection_threshold, distance=5)
  all_classes_peak_mask = np.zeros(all_classes_max.shape, dtype = 'bool')
  all_classes_peak_mask[all_classes_peaks] = True
  all_classes_peak_mask = np.expand_dims(all_classes_peak_mask, -1) #mask: look at peaks taken across all classes
  
  preds = []
  for i in range(np.shape(predictions)[-1]):
    x = predictions[:,i]
    peaks, _ = find_peaks(x, height=args.detection_threshold, distance=5)
    anchor_peak_preds = np.zeros(x.shape, dtype='bool')
    anchor_peak_preds[peaks] = True
    anchor_peak_is_max_mask = x == all_classes_max # mask: enforce class peaks are really maxima
    anchor_peak_preds = anchor_peak_preds * anchor_peak_is_max_mask 
    preds.append(anchor_peak_preds)
    
  preds = np.stack(preds, axis = -1)
  preds = preds * all_classes_peak_mask

  print(f"Peaks found {np.sum(preds)} boxes")
  
  pred_sr = args.sr // (args.scale_factor * args.prediction_scale_factor)
  anchor_scores = predictions
  
  bboxes, scores, class_idxs = pred2bbox(preds, anchor_scores, regressions, pred_sr)
  
  target_fp = os.path.join(args.experiment_dir, f"peaks_pred_{fn}.txt")
  st = bbox2raven(bboxes, class_idxs, args.label_set)
  write_tsv(target_fp, st)
  
  return target_fp
  
def get_metrics(predictions_fp, annotations_fp, args):
  c = Clip(label_set=args.label_set, unknown_label=args.unknown_label)
  
  c.load_predictions(predictions_fp)
  c.load_annotations(annotations_fp, label_mapping = args.label_mapping)
  
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
  class_labels = sorted(metrics[fps[0]][iou_thresholds[0]].keys())
  
  overall = {iou_thresh : { l: {'TP' : 0, 'FP' : 0, 'FN' : 0} for l in class_labels} for iou_thresh in iou_thresholds}
  
  for fp in fps:
    for iou_thresh in iou_thresholds:
      for l in class_labels:
        counts = metrics[fp][iou_thresh][l]
        overall[iou_thresh][l]['TP'] += counts['TP']
        overall[iou_thresh][l]['FP'] += counts['FP']
        overall[iou_thresh][l]['FN'] += counts['FN']
      
  for iou_thresh in iou_thresholds:
    for l in class_labels:
      tp = overall[iou_thresh][l]['TP']
      fp = overall[iou_thresh][l]['FP']
      fn = overall[iou_thresh][l]['FN']

      if tp + fp == 0:
        prec = 1
      else:
        prec = tp / (tp + fp)
      overall[iou_thresh][l]['precision'] = prec

      if tp + fn == 0:
        rec = 1
      else:
        rec = tp / (tp + fn)
      overall[iou_thresh][l]['recall'] = rec

      if prec + rec == 0:
        f1 = 0
      else:
        f1 = 2*prec*rec / (prec + rec)
      overall[iou_thresh][l]['f1'] = f1
  
  return overall

def predict_and_evaluate(model, dataloader_dict, args):
  metrics = {}
  for fn in dataloader_dict:
    predictions, regressions = generate_predictions(model, dataloader_dict[fn], args)
    predictions_fp = export_to_selection_table(predictions, regressions, fn, args)
    annotations_fp = os.path.join(args.annotation_selection_tables_dir, f"{fn}.txt")
    metrics[fn] = get_metrics(predictions_fp, annotations_fp, args)
  
  summary = summarize_metrics(metrics)
  metrics['summary'] = summary
  return metrics



# def soft_nms(bbox_preds, bbox_scores, sigma=0.5, thresh=0.001):
#     """
#     Modified from https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py

#     Reference
#     https://arxiv.org/abs/1704.04503

#     Build a Soft NMS algorithm.
#     # Augments
#         bbox_preds: numpy array
#             shape=(num_bboxes, 2)

#         bbox_scores: numpy array
#             shape=(num_bboxes,)

#         sigma:       variance of Gaussian function

#         thresh:      score thresh

#     # Return
#         the index of the selected boxes
#     """

#     bbox_preds0 = bbox_preds
#     bbox_preds = bbox_preds.copy()
#     bbox_scores = bbox_scores.copy()

#     # Indexes concatenate boxes with the last column
#     N = bbox_preds.shape[0]

#     assert (bbox_scores.shape[0] == N)

#     if N == 0:
#         return None, []

#     bbox_preds = np.concatenate((bbox_preds, np.arange(0, N)[:, None]), axis=1)

#     # The order of boxes coordinate is [start, end]
#     start = bbox_preds[:, 0]
#     end = bbox_preds[:, 1]

#     scores = bbox_scores

#     areas = end-start

#     for i in range(N):
#         # intermediate parameters for later parameters exchange
#         tscore = scores[i].copy()
#         pos = i + 1

#         if i != N - 1:
#             rel_maxpos = np.argmax(scores[pos:], axis=0)
#             maxpos = pos + rel_maxpos
#             maxscore = scores[maxpos]

#             if tscore < maxscore:
#                 bbox_preds[i], bbox_preds[maxpos] = bbox_preds[maxpos].copy(), bbox_preds[i].copy()
#                 scores[i], scores[maxpos] = scores[maxpos].copy(), scores[i].copy()
#                 areas[i], areas[maxpos] = areas[maxpos].copy(), areas[i].copy()

#         # IoU calculate
#         xx1 = np.maximum(bbox_preds[i, 0], bbox_preds[pos:, 0])
#         xx2 = np.minimum(bbox_preds[i, 1], bbox_preds[pos:, 1])

#         inter = np.maximum(0.0, xx2 - xx1)
        
#         ovr = np.divide(inter, (areas[i] + areas[pos:] - inter))

#         # Gaussian decay
#         weight = np.exp(-(ovr * ovr) / sigma)
#         scores[pos:] = weight * scores[pos:]

#     # select the boxes and keep the corresponding indexes
#     keep_indices = bbox_preds[:, 2][scores > thresh].astype(int)
#     new_bbox_preds = bbox_preds0[keep_indices, :2]

#     return new_bbox_preds, keep_indices


# def nms(bbox_preds, bbox_scores, iou_thresh=0.5):
#     """
#     Modified from https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py

#     Reference
#     https://arxiv.org/abs/1704.04503

#     Build a Soft NMS algorithm.
#     # Augments
#         bbox_preds: numpy array
#             shape=(num_bboxes, 2)

#         bbox_scores: numpy array
#             shape=(num_bboxes,)

#         sigma:       variance of Gaussian function

#         thresh:      score thresh

#     # Return
#         the index of the selected boxes
#     """

#     bbox_preds0 = bbox_preds
#     bbox_preds = bbox_preds.copy()
#     bbox_scores = bbox_scores.copy()

#     # Indexes concatenate boxes with the last column
#     N = bbox_preds.shape[0]

#     assert (bbox_scores.shape[0] == N)

#     if N == 0:
#         return None, []

#     bbox_preds = np.concatenate((bbox_preds, np.arange(0, N)[:, None]), axis=1)

#     # The order of boxes coordinate is [start, end]
#     start = bbox_preds[:, 0]
#     end = bbox_preds[:, 1]

#     scores = bbox_scores

#     areas = end-start

#     for i in range(N):
#         # intermediate parameters for later parameters exchange
#         tscore = scores[i].copy()
#         pos = i + 1

#         if i != N - 1:
#             rel_maxpos = np.argmax(scores[pos:], axis=0)
#             maxpos = pos + rel_maxpos
#             maxscore = scores[maxpos]

#             if tscore < maxscore:
#                 bbox_preds[i], bbox_preds[maxpos] = bbox_preds[maxpos].copy(), bbox_preds[i].copy()
#                 scores[i], scores[maxpos] = scores[maxpos].copy(), scores[i].copy()
#                 areas[i], areas[maxpos] = areas[maxpos].copy(), areas[i].copy()

#         # IoU calculate
#         xx1 = np.maximum(bbox_preds[i, 0], bbox_preds[pos:, 0])
#         xx2 = np.minimum(bbox_preds[i, 1], bbox_preds[pos:, 1])

#         inter = np.maximum(0.0, xx2 - xx1)

#         ovr = np.divide(inter, (areas[i] + areas[pos:] - inter))

#         # Gaussian decay
#         weight = (ovr < iou_thresh)
#         scores[pos:] = weight * scores[pos:]

#     # select the boxes and keep the corresponding indexes
#     keep_indices = bbox_preds[:, 2][scores > 0].astype(int)
#     new_bbox_preds = bbox_preds0[keep_indices, :2]

#     return new_bbox_preds, keep_indices
