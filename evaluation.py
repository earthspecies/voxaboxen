import numpy as np
import csv
import torch
import os
import tqdm
from raven_utils import Clip
from model import preprocess_and_augment
from scipy.signal import find_peaks
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path

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


def generate_predictions(model, single_clip_dataloader, args):
  model = model.to(device)
  model.eval()
  
  all_predictions = []
  all_regressions = []
  with torch.no_grad():
    for i, X in tqdm.tqdm(enumerate(single_clip_dataloader)):
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

def get_confusion_matrix(predictions_fp, annotations_fp, args):
  c = Clip(label_set=args.label_set, unknown_label=args.unknown_label)
  
  c.load_predictions(predictions_fp)
  c.load_annotations(annotations_fp, label_mapping = args.label_mapping)
  
  confusion_matrix = {}
  
  for iou_thresh in [0.2, 0.5, 0.8]:
    c.compute_matching(IoU_minimum = iou_thresh)
    confusion_matrix[iou_thresh], confusion_matrix_labels = c.confusion_matrix()
  
  return confusion_matrix, confusion_matrix_labels

def summarize_metrics(metrics):
  # metrics (dict) : {fp : fp_metrics}
  # where
  # fp_metrics (dict) : {iou_thresh : {'TP': int, 'FP' : int, 'FN' : int}}
  
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

def plot_confusion_matrix(data, label_names, target_dir, name=""):   
    fig = plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    sns.heatmap(data, annot=True, fmt='d', cmap = 'magma', cbar = True, ax = ax)
    ax.set_title('Confusion Matrix')
    ax.set_yticks([i + 0.5 for i in range(len(label_names))])
    ax.set_yticklabels(label_names, rotation = 0)
    ax.set_xticks([i + 0.5 for i in range(len(label_names))])
    ax.set_xticklabels(label_names, rotation = -15)
    ax.set_ylabel('Prediction')
    ax.set_xlabel('Annotation')
    plt.title(name)
    
    plt.savefig(Path(target_dir, f"{name}_confusion_matrix.png"))

def summarize_confusion_matrix(confusion_matrix, confusion_matrix_labels):
  # confusion_matrix (dict) : {fp : fp_cm}
  # where
  # fp_cm (dict) : {iou_thresh : numpy array}
  
  fps = sorted(confusion_matrix.keys())
  iou_thresholds = sorted(confusion_matrix[fps[0]].keys())
  l = len(confusion_matrix_labels)
  
  overall = {iou_thresh : np.zeros((l, l)) for iou_thresh in iou_thresholds}
  
  for fp in fps:
    for iou_thresh in iou_thresholds:
      overall[iou_thresh] += confusion_matrix[fp][iou_thresh]
  
  return overall, confusion_matrix_labels

def predict_and_evaluate(model, dataloader_dict, args, save = True):
  metrics = {}
  confusion_matrix = {}
  for fn in dataloader_dict:
    predictions, regressions = generate_predictions(model, dataloader_dict[fn], args)
    predictions_fp = export_to_selection_table(predictions, regressions, fn, args)
    annotations_fp = dataloader_dict[fn].dataset.annot_fp
    metrics[fn] = get_metrics(predictions_fp, annotations_fp, args)
    confusion_matrix[fn], confusion_matrix_labels = get_confusion_matrix(predictions_fp, annotations_fp, args)
  
  # summarize and save metrics
  summary = summarize_metrics(metrics)
  metrics['summary'] = summary
  if save:
    metrics_fp = os.path.join(args.experiment_dir, 'metrics.yaml')
    with open(metrics_fp, 'w') as f:
      yaml.dump(metrics, f)
  
  # summarize and save confusion matrix
  confusion_matrix_summary, confusion_matrix_labels = summarize_confusion_matrix(confusion_matrix, confusion_matrix_labels)
  if save:
    for key in confusion_matrix_summary:
      plot_confusion_matrix(confusion_matrix_summary[key].astype(int), confusion_matrix_labels, args.experiment_dir, name=f"iou_{key}")  
  
  return metrics, confusion_matrix_summary
