import numpy as np
import csv
import torch
import os
import tqdm
from scipy.signal import find_peaks
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from source.evaluation.raven_utils import Clip
from source.model.model import rms_and_mixup

device = "cuda" if torch.cuda.is_available() else "cpu"

def pred2bbox(detection_peaks, detection_probs, durations, class_idxs, class_probs, pred_sr):
    '''
    detection_peaks, detection_probs, durations, class_idxs, class_probs :
        shape=(num_frames,)

    pred_sr:
        prediction sampling rate in Hz

    '''
    detection_peaks = detection_peaks / pred_sr
    bboxes = []
    detection_probs_sub = []
    class_idxs_sub = []
    class_probs_sub = []
            
    for i in range(len(detection_peaks)):
        duration = durations[i]
        start = detection_peaks[i]
                   
        if duration <= 0:
          continue
         
        bbox = [start, start+duration]
        bboxes.append(bbox)
        
        detection_probs_sub.append(detection_probs[i])
        class_idxs_sub.append(class_idxs[i])
        class_probs_sub.append(class_probs[i])
        
    return np.array(bboxes), np.array(detection_probs_sub), np.array(class_idxs_sub), np.array(class_probs_sub)
    
def bbox2raven(bboxes, class_idxs, label_set, detection_probs, class_probs, unknown_label):
    '''
    output bounding boxes to a selection table

    out_fp:
        output file path

    bboxes: numpy array
        shape=(num_bboxes, 2)
        
    class_idxs: numpy array
        shape=(num_bboxes,)

    label_set: list
    
    detection_probs: numpy array
        shape =(num_bboxes,)
        
    class_probs: numpy array
        shape = (num_bboxes,)
        
    unknown_label: str

    '''
    if bboxes is None:
      return [['Begin Time (s)', 'End Time (s)', 'Annotation', 'Detection Prob', 'Class Prob']]

    columns = ['Begin Time (s)', 'End Time (s)', 'Annotation', 'Detection Prob', 'Class Prob']
    
    
    def label_idx_to_label(i):
      if i==-1:
        return unknown_label
      else:
        return label_set[i]
        
    out_data = [[bbox[0], bbox[1], label_idx_to_label(int(c)), dp, cp] for bbox, c, dp, cp in zip(bboxes, class_idxs, detection_probs, class_probs)]
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


def generate_predictions(model, single_clip_dataloader, args, verbose = True):
  assert single_clip_dataloader.dataset.clip_hop == args.clip_duration/2, "For inference, clip hop is assumed to be equal to half clip duration"

  model = model.to(device)
  model.eval()
  
  all_detections = []
  all_regressions = []
  all_classifications = []
  
  if verbose:
    iterator = tqdm.tqdm(enumerate(single_clip_dataloader), total=len(single_clip_dataloader))
  else:
    iterator = enumerate(single_clip_dataloader)
  
  with torch.no_grad():
    for i, X in iterator:
      X = X.to(device = device, dtype = torch.float)
      X, _, _, _ = rms_and_mixup(X, None, None, None, False, args)
      
      detection, regression, classification = model(X)
      classification = torch.nn.functional.softmax(classification, dim=-1)
      
      all_detections.append(detection)
      all_regressions.append(regression)
      all_classifications.append(classification)
      
    all_detections = torch.cat(all_detections)
    all_regressions = torch.cat(all_regressions)
    all_classifications = torch.cat(all_classifications)

    # we use half overlapping windows, need to throw away boundary predictions
    # See get_val_dataloader and get_test_dataloader in data.py
    
    ######## Todo: Need better checking that preds are the correct dur    
    assert all_detections.size(dim=1) % 2 == 0
    first_quarter_window_dur_samples=all_detections.size(dim=1)//4
    last_quarter_window_dur_samples=(all_detections.size(dim=1)//2)-first_quarter_window_dur_samples
    
    # assemble detections
    beginning_bit = all_detections[0,:first_quarter_window_dur_samples]
    end_bit = all_detections[-1,-last_quarter_window_dur_samples:]
    detections_clipped = all_detections[:,first_quarter_window_dur_samples:-last_quarter_window_dur_samples]
    all_detections = torch.reshape(detections_clipped, (-1,))
    all_detections = torch.cat([beginning_bit, all_detections, end_bit])
    
    # assemble regressions
    beginning_bit = all_regressions[0,:first_quarter_window_dur_samples]
    end_bit = all_regressions[-1,-last_quarter_window_dur_samples:]
    regressions_clipped = all_regressions[:,first_quarter_window_dur_samples:-last_quarter_window_dur_samples]
    all_regressions = torch.reshape(regressions_clipped, (-1,))
    all_regressions = torch.cat([beginning_bit, all_regressions, end_bit])
    
    # assemble classifications
    beginning_bit = all_classifications[0,:first_quarter_window_dur_samples, :]
    end_bit = all_classifications[-1,-last_quarter_window_dur_samples:, :]
    classifications_clipped = all_classifications[:,first_quarter_window_dur_samples:-last_quarter_window_dur_samples,:]
    all_classifications = torch.reshape(classifications_clipped, (-1, classifications_clipped.size(-1)))
    all_classifications = torch.cat([beginning_bit, all_classifications, end_bit])
    
  return all_detections.detach().cpu().numpy(), all_regressions.detach().cpu().numpy(), all_classifications.detach().cpu().numpy()

def generate_features(model, single_clip_dataloader, args, verbose = True):
  model = model.to(device)
  model.eval()
  
  all_features = []
  
  if verbose:
    iterator = tqdm.tqdm(enumerate(single_clip_dataloader), total=len(single_clip_dataloader))
  else:
    iterator = enumerate(single_clip_dataloader)
  
  with torch.no_grad():
    for i, X in iterator:
      X = X.to(device = device, dtype = torch.float)
      X, _, _, _ = rms_and_mixup(X, None, None, None, False, args)
      features = model.generate_features(X)
      all_features.append(features)
    all_features = torch.cat(all_features)
    
    ######## Need better checking that features are the correct dur    
    assert all_features.size(dim=1) % 2 == 0
    first_quarter_window_dur_samples=all_features.size(dim=1)//4
    last_quarter_window_dur_samples=(all_features.size(dim=1)//2)-first_quarter_window_dur_samples
    
    # assemble features
    beginning_bit = all_features[0,:first_quarter_window_dur_samples,:]
    end_bit = all_features[-1,-last_quarter_window_dur_samples:,:]
    features_clipped = all_features[:,first_quarter_window_dur_samples:-last_quarter_window_dur_samples,:]
    all_features = torch.reshape(features_clipped, (-1, features_clipped.size(-1)))
    all_features = torch.cat([beginning_bit, all_features, end_bit])
    
  return all_features.detach().cpu().numpy()

def export_to_selection_table(detections, regressions, classifications, fn, args, verbose=True, target_dir=None, detection_threshold = 0.5, classification_threshold = 0):
    
  if target_dir is None:
    target_dir = args.experiment_output_dir  

#   Debugging
#
#   target_fp = os.path.join(target_dir, f"detections_{fn}.npy")
#   np.save(target_fp, detections)
  
#   target_fp = os.path.join(target_dir, f"regressions_{fn}.npy")
#   np.save(target_fp, regressions)
  
#   target_fp = os.path.join(target_dir, f"classifications_{fn}.npy")
#   np.save(target_fp, classifications)
  
  ## peaks  
  detection_peaks, properties = find_peaks(detections, height = detection_threshold, distance=args.peak_distance)
  detection_probs = properties['peak_heights']
    
  ## regressions and classifications
  durations = []
  class_idxs = []
  class_probs = []
  
  for i in detection_peaks:
    dur = regressions[i]
    durations.append(dur)
    
    c = np.argmax(classifications[i,:])    
    p = classifications[i,c]
    
    if p < classification_threshold:
      c = -1
    
    class_idxs.append(c)
    class_probs.append(p)
        
  durations = np.array(durations)
  class_idxs = np.array(class_idxs)
  class_probs = np.array(class_probs)
    
  if verbose:
    print(f"Peaks found {len(detection_peaks)} boxes")
  
  pred_sr = args.sr // (args.scale_factor * args.prediction_scale_factor)
  
  bboxes, detection_probs, class_idxs, class_probs = pred2bbox(detection_peaks, detection_probs, durations, class_idxs, class_probs, pred_sr)
  
  target_fp = os.path.join(target_dir, f"peaks_pred_{fn}.txt")
    
  st = bbox2raven(bboxes, class_idxs, args.label_set, detection_probs, class_probs, args.unknown_label)
  write_tsv(target_fp, st)
  
  return target_fp
  
def get_metrics(predictions_fp, annotations_fp, args, iou, class_threshold):
  c = Clip(label_set=args.label_set, unknown_label=args.unknown_label)
  
  c.load_predictions(predictions_fp)
  c.threshold_class_predictions(class_threshold)
  c.load_annotations(annotations_fp, label_mapping = args.label_mapping)
  
  metrics = {}
  
  c.compute_matching(IoU_minimum = iou)
  metrics = c.evaluate()
  
  return metrics

def get_confusion_matrix(predictions_fp, annotations_fp, args, iou, class_threshold):
  c = Clip(label_set=args.label_set, unknown_label=args.unknown_label)
  
  c.load_predictions(predictions_fp)
  c.threshold_class_predictions(class_threshold)
  c.load_annotations(annotations_fp, label_mapping = args.label_mapping)
  
  confusion_matrix = {}
  
  c.compute_matching(IoU_minimum = iou)
  confusion_matrix, confusion_matrix_labels = c.confusion_matrix()
  
  return confusion_matrix, confusion_matrix_labels

def summarize_metrics(metrics):
  # metrics (dict) : {fp : fp_metrics}
  # where
  # fp_metrics (dict) : {class_label: {'TP': int, 'FP' : int, 'FN' : int}}
  
  fps = sorted(metrics.keys())
  class_labels = sorted(metrics[fps[0]].keys())
  
  overall = { l: {'TP' : 0, 'FP' : 0, 'FN' : 0} for l in class_labels}
  
  for fp in fps:
    for l in class_labels:
      counts = metrics[fp][l]
      overall[l]['TP'] += counts['TP']
      overall[l]['FP'] += counts['FP']
      overall[l]['FN'] += counts['FN']
      
  for l in class_labels:
    tp = overall[l]['TP']
    fp = overall[l]['FP']
    fn = overall[l]['FN']

    if tp + fp == 0:
      prec = 1
    else:
      prec = tp / (tp + fp)
    overall[l]['precision'] = prec

    if tp + fn == 0:
      rec = 1
    else:
      rec = tp / (tp + fn)
    overall[l]['recall'] = rec

    if prec + rec == 0:
      f1 = 0
    else:
      f1 = 2*prec*rec / (prec + rec)
    overall[l]['f1'] = f1
  
  return overall

def macro_metrics(summary):
  # summary (dict) : {class_label: {'f1' : float, 'precision' : float, 'recall' : float, 'TP': int, 'FP' : int, 'FN' : int}}
  
  metrics = ['f1', 'precision', 'recall']
  
  macro = {}
  
  for metric in metrics:

    e = []
    for l in summary:
        m = summary[l][metric]
        e.append(m)
    macro[metric] = float(np.mean(e))
  
  return macro

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
    
    plt.savefig(os.path.join(target_dir, f"{name}_confusion_matrix.svg"))
    plt.close()


def summarize_confusion_matrix(confusion_matrix, confusion_matrix_labels):
  # confusion_matrix (dict) : {fp : fp_cm}
  # where
  # fp_cm  : numpy array
  
  fps = sorted(confusion_matrix.keys())
  l = len(confusion_matrix_labels)
  
  overall = np.zeros((l, l))
  
  for fp in fps:
    overall += confusion_matrix[fp]
  
  return overall, confusion_matrix_labels

def predict_and_generate_manifest(model, dataloader_dict, args, verbose = True):
  fns = []
  predictions_fps = []
  annotations_fps = []
                               
  for fn in dataloader_dict:
    detections, regressions, classifications = generate_predictions(model, dataloader_dict[fn], args, verbose=verbose)
    
    predictions_fp = export_to_selection_table(detections, regressions, classifications, fn, args, verbose = verbose, detection_threshold = args.detection_threshold)
    
    annotations_fp = dataloader_dict[fn].dataset.annot_fp
    
    fns.append(fn)
    predictions_fps.append(predictions_fp)
    annotations_fps.append(annotations_fp)
    
  manifest = pd.DataFrame({'filename' : fns, 'predictions_fp' : predictions_fps, 'annotations_fp' : annotations_fps})
  return manifest
                                  
def evaluate_based_on_manifest(manifest, args, output_dir = None, iou = 0.5, class_threshold = 0.0):
  
  metrics = {}
  confusion_matrix = {}
  
  for i, row in manifest.iterrows():
    fn = row['filename']
    predictions_fp = row['predictions_fp']
    annotations_fp = row['annotations_fp']
  
    metrics[fn] = get_metrics(predictions_fp, annotations_fp, args, iou, class_threshold)
    confusion_matrix[fn], confusion_matrix_labels = get_confusion_matrix(predictions_fp, annotations_fp, args, iou, class_threshold)
    
  if output_dir is not None:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  
  # summarize and save metrics
  summary = summarize_metrics(metrics)
  metrics['summary'] = summary
  macro = macro_metrics(summary)
  metrics['macro'] = macro
  if output_dir is not None:
    metrics_fp = os.path.join(output_dir, f'metrics_iou_{iou}_class_threshold_{class_threshold}.yaml')
    with open(metrics_fp, 'w') as f:
      yaml.dump(metrics, f)
  
  # summarize and save confusion matrix
  confusion_matrix_summary, confusion_matrix_labels = summarize_confusion_matrix(confusion_matrix, confusion_matrix_labels)
  if output_dir is not None:
    plot_confusion_matrix(confusion_matrix_summary.astype(int), confusion_matrix_labels, output_dir, name=f"cm_iou_{iou}_class_threshold_{class_threshold}")  
  
  return metrics, confusion_matrix_summary
