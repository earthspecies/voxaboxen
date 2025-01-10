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

from voxaboxen.evaluation.raven_utils import Clip
from voxaboxen.model.model import rms_and_mixup
from voxaboxen.evaluation.nms import nms, soft_nms
from voxaboxen.evaluation.conf_mats import get_confusion_matrix, summarize_confusion_matrix, plot_confusion_matrix
from fn_profiling import profile_lines
plt.switch_backend('agg')

device = "cuda" if torch.cuda.is_available() else "cpu"

def f1_from_counts(tp, fp, fn):
    prec = tp/(tp+fp+1e-8)
    rec = tp/(tp+fn+1e-8)
    f1 = 2*prec*rec / (prec+rec+1e-8)
    return {'prec': prec, 'rec':rec, 'f1':f1}

def macro_micro_metrics(summary, unknown_label="Unknown"):
    """summary (dict) : {class_label: {'f1' : float, 'precision' : float, 'recall' : float, 'f1_seg' : float, 'precision_seg' : float, 'recall_seg' : float, 'TP': int, 'FP' : int, 'FN' : int, TP_seg': int, 'FP_seg' : int, 'FN_seg' : int}}"""

    metrics = ['f1', 'precision', 'recall', 'f1_seg', 'precision_seg', 'recall_seg']
    macro = {}

    for metric in metrics:

        e = []
        for l in summary:
            if l == unknown_label:
                continue
            m = summary[l][metric]
            e.append(m)
        macro[metric] = float(np.mean(e))

    tp = sum(v['TP'] for v in summary.values())
    fp = sum(v['FP'] for v in summary.values())
    fn = sum(v['FN'] for v in summary.values())
    micro = f1_from_counts(tp, fp, fn)
    tp = sum(v['TP_seg'] for v in summary.values())
    fp = sum(v['FP_seg'] for v in summary.values())
    fn = sum(v['FN_seg'] for v in summary.values())
    seg_micro = f1_from_counts(tp, fp, fn)
    seg_micro = {f'{k}_seg': v for k,v in seg_micro.items()}
    micro.update(seg_micro)

    return macro, micro

def get_metrics(predictions_fp, annotations_fp, iou, class_threshold, duration, label_mapping, unknown_label):
    label_set = list(label_mapping.keys())
    c = Clip(label_set=label_set, unknown_label=unknown_label)
    c.duration = duration
    c.load_predictions(predictions_fp)
    c.threshold_class_predictions(class_threshold)
    assert not any(c.predictions['Annotation']=='Unknown')
    c.load_annotations(annotations_fp, label_mapping = label_mapping)

    metrics = {}

    c.compute_matching(IoU_minimum = iou)
    metrics = c.evaluate()

    return metrics

def summarize_metrics(metrics):
    """ metrics (dict) : {fp : fp_metrics},  where
    # fp_metrics (dict) : {class_label: {'TP': int, 'FP' : int, 'FN' : int, 'TP_seg' : int, 'FP_seg' : int, 'FN_seg' : int}}
    """

    fps = sorted(metrics.keys())
    class_labels = sorted(metrics[fps[0]].keys())

    overall = { l: {'TP' : 0, 'FP' : 0, 'FN' : 0, 'TP_seg' : 0, 'FP_seg' : 0, 'FN_seg' : 0} for l in class_labels}

    for fp in fps:
      for l in class_labels:
        counts = metrics[fp][l]
        overall[l]['TP'] += counts['TP']
        overall[l]['FP'] += counts['FP']
        overall[l]['FN'] += counts['FN']
        overall[l]['TP_seg'] += counts['TP_seg']
        overall[l]['FP_seg'] += counts['FP_seg']
        overall[l]['FN_seg'] += counts['FN_seg']

    for l in class_labels:
      tp = overall[l]['TP']
      fp = overall[l]['FP']
      fn = overall[l]['FN']
      tp_seg = overall[l]['TP_seg']
      fp_seg = overall[l]['FP_seg']
      fn_seg = overall[l]['FN_seg']

      if tp + fp == 0:
        prec = 1
      else:
        prec = tp / (tp + fp)
      overall[l]['precision'] = prec

      if tp_seg + fp_seg == 0:
        prec_seg = 1
      else:
        prec_seg = tp_seg / (tp_seg + fp_seg)
      overall[l]['precision_seg'] = prec_seg

      if tp + fn == 0:
        rec = 1
      else:
        rec = tp / (tp + fn)
      overall[l]['recall'] = rec

      if tp_seg + fn_seg == 0:
        rec_seg = 1
      else:
        rec_seg = tp_seg / (tp_seg + fn_seg)
      overall[l]['recall_seg'] = rec_seg

      if prec + rec == 0:
        f1 = 0
      else:
        f1 = 2*prec*rec / (prec + rec)
      overall[l]['f1'] = f1

      if prec_seg + rec_seg == 0:
        f1_seg = 0
      else:
        f1_seg = 2*prec_seg*rec_seg / (prec_seg + rec_seg)
      overall[l]['f1_seg'] = f1_seg

    return overall

def generate_predictions(model, single_clip_dataloader, args, verbose=True):
    assert single_clip_dataloader.dataset.clip_hop == args.clip_duration/2, "For inference, clip hop is assumed to be equal to half clip duration"

    model = model.to(device)
    model.eval()

    all_detections = []
    all_regressions = []
    all_classifs = []
    all_rev_detections = []
    all_rev_regressions = []
    all_rev_classifs = []

    if verbose:
      iterator = tqdm.tqdm(enumerate(single_clip_dataloader), total=len(single_clip_dataloader))
    else:
      iterator = enumerate(single_clip_dataloader)

    with torch.no_grad():
      for i, X in iterator:
        X = X.to(device = device, dtype = torch.float)
        X, _, _, _ = rms_and_mixup(X, None, None, None, False, args)

        model_outputs = model(X)
        assert isinstance(model_outputs, tuple)
        all_detections.append(model_outputs[0])
        all_regressions.append(model_outputs[1])
        if args.segmentation_based:
            classification=torch.nn.functional.sigmoid(model_outputs[2])
        else:
            classification=torch.nn.functional.softmax(model_outputs[2], dim=-1)
        all_classifs.append(classification)
        if model.is_bidirectional:
            assert all(x is not None for x in model_outputs)
            all_rev_detections.append(model_outputs[3])
            all_rev_regressions.append(model_outputs[4])
            all_rev_classifs.append(model_outputs[5].softmax(-1)) # segmentation-based is not used when bidirectional
        else:
            assert all(x is None for x in model_outputs[3:])

        if args.is_test and i==15:
            break

      all_detections = torch.cat(all_detections)
      all_regressions = torch.cat(all_regressions)
      all_classifs = torch.cat(all_classifs)
      if model.is_bidirectional:
          all_rev_detections = torch.cat(all_rev_detections)
          all_rev_regressions = torch.cat(all_rev_regressions)
          all_rev_classifs = torch.cat(all_rev_classifs)

      ######## Todo: Need better checking that preds are the correct dur
      assert all_detections.size(dim=1) % 2 == 0
      first_quarter_window_dur_samples=all_detections.size(dim=1)//4
      last_quarter_window_dur_samples=(all_detections.size(dim=1)//2)-first_quarter_window_dur_samples

      def assemble(d, r, c):
          """We use half overlapping windows, need to throw away boundary predictions.
          See get_val_dataloader and get_test_dataloader in data.py"""
          # assemble detections
          beginning_d_bit = d[0,:first_quarter_window_dur_samples]
          end_d_bit = d[-1,-last_quarter_window_dur_samples:]
          d_clipped = d[:,first_quarter_window_dur_samples:-last_quarter_window_dur_samples]
          middle_d_bit = torch.reshape(d_clipped, (-1,))
          assembled_d = torch.cat([beginning_d_bit, middle_d_bit, end_d_bit])

          # assemble regressions
          beginning_r_bit = r[0,:first_quarter_window_dur_samples]
          end_r_bit = r[-1,-last_quarter_window_dur_samples:]
          r_clipped = r[:,first_quarter_window_dur_samples:-last_quarter_window_dur_samples]
          middle_r_bit = torch.reshape(r_clipped, (-1,))
          assembled_r = torch.cat([beginning_r_bit, middle_r_bit, end_r_bit])

          # assemble classifs
          beginning_c_bit = c[0,:first_quarter_window_dur_samples, :]
          end_c_bit = c[-1,-last_quarter_window_dur_samples:, :]
          c_clipped = c[:,first_quarter_window_dur_samples:-last_quarter_window_dur_samples,:]
          middle_c_bit = torch.reshape(c_clipped, (-1, c_clipped.size(-1)))
          assembled_c = torch.cat([beginning_c_bit, middle_c_bit, end_c_bit])
          return assembled_d.detach().cpu().numpy(), assembled_r.detach().cpu().numpy(), assembled_c.detach().cpu().numpy(),

      assembled_dets, assembled_regs, assembled_classifs = assemble(all_detections, all_regressions, all_classifs)
      if model.is_bidirectional:
          assembled_rev_dets, assembled_rev_regs, assembled_rev_classifs = assemble(all_rev_detections, all_rev_regressions, all_rev_classifs)
      else:
          assembled_rev_dets = assembled_rev_regs = assembled_rev_classifs = None

      return assembled_dets, assembled_regs, assembled_classifs, assembled_rev_dets, assembled_rev_regs, assembled_rev_classifs

def pred2bbox(detection_peaks, detection_probs, durations, class_idxs, class_probs, pred_sr, is_rev):
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
        if duration <= 0:
          continue

        if is_rev:
            end = detection_peaks[i]
            bbox = [end-duration, end]
        else:
            start = detection_peaks[i]
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

def fill_holes(m, max_hole):
    stops = m[:-1] * ~m[1:]
    stops = np.where(stops)[0]

    for stop in stops:
        look_forward = m[stop+1:stop+1+max_hole]
        if np.any(look_forward):
            next_start = np.amin(np.where(look_forward)[0]) + stop + 1
            m[stop : next_start] = True

    return m

def delete_short(m, min_pos):
    starts = m[1:] * ~m[:-1]

    starts = np.where(starts)[0] + 1
    if m[0]:
        starts = np.append(starts, 0)

    clips = []

    for start in starts:
        look_forward = m[start:]
        ends = np.where(~look_forward)[0]
        if len(ends)>0:
            clips.append((start, start+np.amin(ends)))
        else:
            clips.append((start, len(m)-1))

    m = np.zeros_like(m).astype(bool)
    for clip in clips:
        if clip[1] - clip[0] >= min_pos:
            m[clip[0]:clip[1]] = True

    return m

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

def select_from_neg_idxs(df, neg_idxs):
    masked = df.loc[~df.index.isin(neg_idxs)]
    #bool_mask = [i not in neg_idxs for i in range(len(df))]
    #masked2 = df.loc[bool_mask]
    #assert (masked==masked2).all().all()
    return masked

def combine_fwd_bck_preds(target_dir, fn, comb_discard_threshold, det_thresh):
    fwd_preds_fp = os.path.join(target_dir, f'peaks_pred_{fn}-detthresh{det_thresh}-fwd.txt')
    bck_preds_fp = os.path.join(target_dir, f'peaks_pred_{fn}-detthresh{det_thresh}-bck.txt')
    comb_preds_fp = os.path.join(target_dir, f'peaks_pred_{fn}-detthresh{det_thresh}-comb.txt')
    match_preds_fp = os.path.join(target_dir, f'peaks_pred_{fn}-detthresh{det_thresh}-match.txt')
    fwd_preds = pd.read_csv(fwd_preds_fp, sep='\t')
    bck_preds = pd.read_csv(bck_preds_fp, sep='\t')

    c = Clip()
    c.load_annotations(fwd_preds_fp)
    c.load_predictions(bck_preds_fp)
    c.compute_matching(IoU_minimum=0)
    match_preds_list = []
    for fp, bp in c.matching:
        match_pred = fwd_preds.loc[fp].copy()
        bck_pred = bck_preds.iloc[bp]
        bp_end_time = bck_pred['End Time (s)']
        match_pred['End Time (s)'] = bp_end_time
        # Sorta like assuming forward and back predictions are independent, gives a high prob for the matched predictions
        match_pred['Detection Prob'] = 1 - (1-match_pred['Detection Prob'])*(1-bck_pred['Detection Prob'])
        match_preds_list.append(match_pred)

    match_preds = pd.DataFrame(match_preds_list, columns=fwd_preds.columns)

    # Include the union of all predictions that weren't part of the matching
    fwd_matched_idxs = [m[0] for m in c.matching]
    bck_matched_idxs = [m[1] for m in c.matching]
    fwd_unmatched = select_from_neg_idxs(fwd_preds, fwd_matched_idxs)
    bck_unmatched = select_from_neg_idxs(bck_preds, bck_matched_idxs)
    to_concat = [x for x in [match_preds, fwd_unmatched, bck_unmatched] if x.shape[0]>0]
    comb_preds = pd.concat(to_concat) if len(to_concat)>0 else fwd_preds
    assert len(comb_preds) == len(fwd_preds) + len(bck_preds) - len(c.matching)

    # Finally, keep only predictions above a threshold, this will include almost all matches
    comb_preds = comb_preds.loc[comb_preds['Detection Prob']>comb_discard_threshold]
    comb_preds.sort_values('Begin Time (s)')
    comb_preds.index = list(range(len(comb_preds)))

    comb_preds.to_csv(comb_preds_fp, sep='\t', index=False)
    match_preds.to_csv(match_preds_fp, sep='\t', index=False)
    return comb_preds_fp, match_preds_fp

def export_to_selection_table(detections, regressions, classifications, fn, args, is_bck, verbose=True, target_dir=None, detection_threshold = 0.5, classification_threshold = 0):
    if hasattr(args, "bidirectional") and args.bidirectional:
      if is_bck:
        fn += '-bck'
      else:
        fn += '-fwd'

    if target_dir is None:
      target_dir = args.experiment_output_dir

    if hasattr(args, "segmentation_based") and args.segmentation_based:
      pred_sr = args.sr // (args.scale_factor * args.prediction_scale_factor)
      bboxes = []
      det_probs = []
      class_idxs = []
      class_probs = []
      for c in range(np.shape(classifications)[1]):
        classifications_sub=classifications[:,c]
        classifications_sub_binary=(classifications_sub>=detection_threshold)
        classifications_sub_binary=fill_holes(classifications_sub_binary,int(args.fill_holes_dur_sec*pred_sr))
        classifications_sub_binary=delete_short(classifications_sub_binary,int(args.delete_short_dur_sec*pred_sr))

        starts = classifications_sub_binary[1:] * ~classifications_sub_binary[:-1]
        starts = np.where(starts)[0] + 1
        if classifications_sub_binary[0]:
            starts = np.append(starts, 0)

        for start in starts:
            look_forward = classifications_sub_binary[start:]
            ends = np.where(~look_forward)[0]
            if len(ends)>0:
                end = start+np.amin(ends)
            else:
                end = len(classifications_sub_binary)-1

            bbox = [start/pred_sr,end/pred_sr]
            bboxes.append(bbox)
            det_probs.append(classifications_sub[start:end].mean())
            class_idxs.append(c)
            class_probs.append(classifications_sub[start:end].mean())

      bboxes=np.array(bboxes)
      det_probs=np.array(det_probs)
      class_idxs=np.array(class_idxs)
      class_probs=np.array(class_probs)

    else:
      ## peaks
      detection_peaks, properties = find_peaks(detections, height = detection_threshold, distance=args.peak_distance)
      det_probs = properties['peak_heights']

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

      pred_sr = args.sr // (args.scale_factor * args.prediction_scale_factor)

      bboxes, det_probs, class_idxs, class_probs = pred2bbox(detection_peaks, det_probs, durations, class_idxs, class_probs, pred_sr, is_bck)

    if args.nms == "soft_nms":
      bboxes, det_probs, class_idxs, class_probs = soft_nms(bboxes, det_probs, class_idxs, class_probs, sigma=args.soft_nms_sigma, thresh=detection_threshold)
    elif args.nms == "nms":
      bboxes, det_probs, class_idxs, class_probs = nms(bboxes, det_probs, class_idxs, class_probs, iou_thresh=args.nms_thresh)

    if verbose:
      print(f"Found {len(det_probs)} boxes")

    target_fp = os.path.join(target_dir, f"peaks_pred_{fn}.txt")

    st = bbox2raven(bboxes, class_idxs, args.label_set, det_probs, class_probs, args.unknown_label)
    write_tsv(target_fp, st)

    return target_fp

def predict_and_generate_manifest(model, dataloader_dict, args, detection_thresholds=None, verbose = True):
    if detection_thresholds is None:
        detection_thresholds = [args.detection_threshold]
    fns = []
    annotations_fps = []
    fwd_predictions_fps = []
    bck_predictions_fps = []
    durations = []
    for i, (fn_base, dloader) in enumerate(dataloader_dict.items()):
        if args.is_test and i==3:
            break
        annotations_fp = dataloader_dict[fn_base].dataset.annot_fp
        fns.append(fn_base)
        annotations_fps.append(annotations_fp)
        fwd_detections, fwd_regressions, fwd_classifications, bck_detections, bck_regressions, bck_classifications = generate_predictions(model, dloader, args, verbose=verbose)
        durations.append(np.shape(fwd_detections)[0]*args.scale_factor/args.sr)

        fwd_predictions_fps_by_thresh = {}
        bck_predictions_fps_by_thresh = {}
        for det_thresh in detection_thresholds:
            fn = f'{fn_base}-detthresh{det_thresh}'
            fwd_predictions_fp = export_to_selection_table(fwd_detections, fwd_regressions, fwd_classifications, fn, args, is_bck=False, verbose=verbose, detection_threshold=det_thresh)
            if model.is_bidirectional:
                assert all(x is not None for x in [bck_detections, bck_classifications, bck_regressions])
                bck_predictions_fp = export_to_selection_table(bck_detections, bck_regressions, bck_classifications, fn, args, is_bck=True, verbose=verbose, detection_threshold=args.detection_threshold)
            else:
                assert all(x is None for x in [bck_detections, bck_classifications, bck_regressions])
                bck_predictions_fp = None

            fwd_predictions_fps_by_thresh[det_thresh] = fwd_predictions_fp
            bck_predictions_fps_by_thresh[det_thresh] = bck_predictions_fp

        fwd_predictions_fps.append(fwd_predictions_fps_by_thresh)
        bck_predictions_fps.append(bck_predictions_fps_by_thresh)

    manifests_by_thresh = {}
    for dt in detection_thresholds:
        fpfps = [x[dt] for x in fwd_predictions_fps]
        bpfps = [x[dt] for x in bck_predictions_fps]
        manifest = pd.DataFrame({'filename' : fns, 'fwd_predictions_fp' : fpfps, 'bck_predictions_fp' : bpfps, 'annotations_fp' : annotations_fps, 'duration_sec' : durations})
        manifests_by_thresh[dt] = manifest
    return manifests_by_thresh

#def evaluate_based_on_manifest(manifest, output_dir, results_dir, iou, class_threshold, label_mapping, unknown_label, det_thresh, comb_discard_threshold=0, bidirectional=False):
def evaluate_based_on_manifest(manifest, output_dir, iou, class_threshold, label_mapping, unknown_label, det_thresh, comb_discard_threshold=0, bidirectional=False, pred_types=None):
    if pred_types is None:
        pred_types = ('fwd', 'bck', 'comb', 'match') if bidirectional else ('fwd',)
    metrics = {p:{} for p in pred_types}

    for i, row in manifest.iterrows():
        fn = row['filename']
        annots_fp = row['annotations_fp']
        duration = row['duration_sec']
        if bidirectional:
            row['comb_predictions_fp'], row['match_predictions_fp'] = combine_fwd_bck_preds(output_dir, fn, comb_discard_threshold=comb_discard_threshold, det_thresh=det_thresh)

        for pred_type in pred_types:
            preds_fp = row[f'{pred_type}_predictions_fp']
            metrics[pred_type][fn] = get_metrics(preds_fp, annots_fp, iou, class_threshold, duration, label_mapping, unknown_label)

    # summarize and save metrics
    conf_mat_summaries = {}
    for pred_type in pred_types:
        summary = summarize_metrics(metrics[pred_type])
        metrics[pred_type]['summary'] = summary
        macro, micro = macro_micro_metrics(summary)
        metrics[pred_type]['macro'], metrics[pred_type]['micro'] = macro, micro
    #if results_dir is not None:
        #os.makedirs(results_dir, exist_ok=True)
        #metrics_fp = os.path.join(results_dir, f'metrics_iou_{iou}_class_threshold_{class_threshold}.yaml')
        #with open(metrics_fp, 'w') as f:
            #yaml.dump(metrics, f)

    return metrics, conf_mat_summaries

def mean_average_precision(manifests_by_thresh, label_mapping, exp_dir, iou=0.5, pred_type='fwd', comb_discard_thresh=0, unknown_label='Unknown', bidirectional=False, best_pred_type='fwd'):
    # first loop through thresholds to gather all results
    scores_by_class = {c:[] for c in label_mapping.keys()}
    experiment_output_dir = os.path.join(exp_dir, 'outputs')
    for det_thresh, test_manifest in manifests_by_thresh.items():
        #results_dir = os.path.join(exp_dir, 'mAP', f'detthresh{det_thresh}')
        test_metrics, test_conf_mats = evaluate_based_on_manifest(test_manifest, output_dir=experiment_output_dir, iou=iou, det_thresh=det_thresh, class_threshold=0.0, comb_discard_threshold=comb_discard_thresh, label_mapping=label_mapping, unknown_label='Unknown', bidirectional=bidirectional, pred_types=(best_pred_type,))
        for c, s in test_metrics[pred_type]['summary'].items():
            scores_by_class[c].append(dict(s, det_thresh=det_thresh))

    # now loop through classes to calculate APs
    ap_by_class = {}
    map_results = {c:{} for c in label_mapping.keys()}
    for c, sweep_ in scores_by_class.items():
        sweep = pd.DataFrame(sweep_)#.sort_values('recall')
        # exclude cases where all TNs because they give weird f-scores
        sweep = sweep.loc[sweep['TP'] + sweep['FP'] + sweep['FN'] != 0]
        precs, recs = list(sweep['precision']), list(sweep['recall'])
        precs = [0.] + precs + [1.]
        recs = [1.] + recs + [0.]
        prec_by_rec = {}
        for r,p in zip(recs, precs):
            if r in prec_by_rec.keys():
                prec_by_rec[r] = max(p, prec_by_rec[r])
            else:
                prec_by_rec[r] = p
        recs, precs = list(prec_by_rec.keys()), list(prec_by_rec.values())

        # 11-point AP computation: https://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
        recs_np = np.array(recs)
        precs_np = np.array(precs)
        auc = 0
        recall_levels = np.linspace(0,1,1001) #np.arange(0,11)/10
        for recall_level in recall_levels:
            p_interp_at_recall_level = np.amax(precs_np[recs_np>=recall_level])
            auc += p_interp_at_recall_level/len(recall_levels)

        if auc==1:
            breakpoint()

        ap_by_class[c] = auc
        map_results[c]['sweep'] = sweep_
        map_results[c]['precs'] = precs
        map_results[c]['recs'] = recs

    # map_score = float(np.array(list(ap_by_class.values())).mean())
    map_score = []
    for c in ap_by_class:
        if c!= unknown_label:
            map_score.append(ap_by_class[c])
    map_score = float(np.array(map_score).mean())

    return map_score, scores_by_class, ap_by_class
