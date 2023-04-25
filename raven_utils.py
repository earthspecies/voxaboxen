import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import yaml
from pathlib import Path
import IPython.display as ipd
import librosa
from tqdm import tqdm
from IPython.core.display import display
import metrics
import warnings

class Clip():
    def __init__(self, label_set = None, unknown_label = None):
        self.sr = None
        self.samples = None
        self.duration = None
        self.annotations = None
        self.predictions = None
        self.matching = None
        self.matched_annotations = None
        self.matched_predictions = None
        self.label_set = label_set
        self.unknown_label = unknown_label
        
    def load_selection_table(self, fp, view = None, label_mapping = None):
        # view (str) : If applicable, Waveform or Spectrogram to avoid double counting
        # label_mapping : dict {old label : new label}. If not None, will drop annotations not in keys of label_mapping
      
      
        annotations = pd.read_csv(fp, delimiter = '\t')
        if view is None and 'View' in annotations:
          views = annotations['View'].unique()
          if len(views)>1:
            warnings.warn(f"I found more than one view in selection table. To avoid double counting, pass view as a parameter. Views found: {view}")
        
        if view is not None:
          annotations = annotations[annotations['View'].str.contains('Waveform')].reset_index()
        
        if label_mapping is not None:
          annotations['Annotation'] = annotations['Annotation'].map(label_mapping)
          annotations = annotations[~pd.isnull(annotations['Annotation'])]
          
        return annotations
        
    def load_audio(self, fp):
        self.samples, self.sr = librosa.load(fp, sr = None)
        self.duration = len(self.samples) / self.sr
        
    def play_audio(self, start_sec, end_sec):
        start_sample = int(self.sr * start_sec)
        end_sample = int(self.sr *end_sec)
        display(ipd.Audio(self.samples[start_sample:end_sample], rate = self.sr))
        
    def load_annotations(self, fp, view = None, label_mapping = None):
        self.annotations = self.load_selection_table(fp, view = view, label_mapping = label_mapping)
        self.annotations['index'] = self.annotations.index
        
    def refine_annotations(self):
        print("Not implemented! Could implement refining annotations by SNR to remove quiet vocs")
        
    def refine_predictions(self):
        print("Not implemented! Could implement refining predictions by SNR to remove quiet vocs")
        
    def load_predictions(self, fp, view = None, label_mapping = None):
        self.predictions = self.load_selection_table(fp, view = view, label_mapping = label_mapping)
        self.predictions['index'] = self.predictions.index
        
    def compute_matching(self, IoU_minimum = 0.5):
        # Bipartite graph matching between predictions and annotations
        # Maximizes the number of matchings with IoU > IoU_minimum
        # Saves a list of indexes of matched pairs
        ref = np.array(self.annotations[['Begin Time (s)', 'End Time (s)']]).T
        est = np.array(self.predictions[['Begin Time (s)', 'End Time (s)']]).T
        self.matching = metrics.match_events(ref, est, min_iou=IoU_minimum, method="fast")
        self.matched_annotations = [p[0] for p in self.matching]
        self.matched_predictions = [p[1] for p in self.matching]
        # print("Computed matching between predictions and annotations based on IoU > %1.3f" % IoU_minimum)
        
    # def refine_matching(self, start_tolerance_sec = 0.2, dur_tolerance_percent = 0.1):
    #     # After matching, we may want to throw out some predictions because their start and end times are too incorrect
    #     count = 0
    #     refined_matchings = []
    #     for match in self.matching:
    #       ref = self.annotations.iloc[match[0]]
    #       ref_start = ref['Begin Time (s)']
    #       ref_dur = ref['End Time (s)'] - ref_start
    #       est = self.predictions.iloc[match[1]]
    #       est_start = est['Begin Time (s)']
    #       est_dur = est['End Time (s)'] - est_start
    #       if np.abs(ref_start-est_start) > start_tolerance_sec:
    #         count +=1
    #         continue
    #       elif np.abs(ref_dur - est_dur) / (ref_dur + 1e-06) > dur_tolerance_percent:
    #         count +=1
    #         continue
    #       else:
    #         refined_matchings.append(match)
    #     self.matching = refined_matchings
    #     self.matched_annotations = [p[0] for p in self.matching]
    #     self.matched_predictions = [p[1] for p in self.matching]
    #     print("Removed %d predictions whose start was off by %1.3f seconds or whose duration was off by %1.3f" % (count, start_tolerance_sec, dur_tolerance_percent))
        
    def evaluate(self):     
      
        if self.label_set is None:
          TP = len(self.matching)
          FP = len(self.predictions) - TP
          FN = len(self.annotations) - TP
          return {'all' : {'TP' : TP, 'FP' : FP, 'FN' : FN}}
        
        else:
          out = {label : {'TP':0, 'FP':0, 'FN' : 0} for label in self.label_set}
          pred_label = np.array(self.predictions['Annotation'])
          annot_label = np.array(self.annotations['Annotation'])
          for p in self.matching:
            annotation = annot_label[p[0]]
            prediction = pred_label[p[1]]
            
            if annotation == prediction:
              out[annotation]['TP'] += 1
            elif self.unknown_label is not None and annotation == self.unknown_label:
              out[prediction]['FP'] -= 1 #adjust FP for unknown labels
              
          for label in self.label_set:
            n_annot = int((annot_label == label).sum())
            n_pred = int((pred_label == label).sum())
            out[label]['FP'] = out[label]['FP'] + n_pred - out[label]['TP']
            out[label]['FN'] = out[label]['FN'] + n_annot - out[label]['TP']
            
          return out
              
    def confusion_matrix(self):
      if self.label_set is None:
        return None
      else:
        confusion_matrix_labels = self.label_set.copy()
        confusion_matrix_labels.append('FP')
        if self.unknown_label is not None:
          confusion_matrix_labels.append(self.unknown_label)
        confusion_matrix_size = len(confusion_matrix_labels)

        confusion_matrix = np.zeros((confusion_matrix_size, confusion_matrix_size))
        cm_fp_idx = confusion_matrix_labels.index('FP')
        
        pred_label = np.array(self.predictions['Annotation'])
        annot_label = np.array(self.annotations['Annotation'])
        
        for p in self.matching:
          annotation = annot_label[p[0]]
          prediction = pred_label[p[1]]
          cm_annot_idx = confusion_matrix_labels.index(annotation)
          cm_pred_idx = confusion_matrix_labels.index(prediction)
          confusion_matrix[cm_pred_idx, cm_annot_idx] += 1

        for label in self.label_set:
          # count false positive detections, regardless of class
          cm_label_idx = confusion_matrix_labels.index(label)
          n_pred = int((pred_label == label).sum())
          n_pred_positive_detections = confusion_matrix.sum(1)[cm_label_idx]
          n_false_detections = n_pred - n_pred_positive_detections
          confusion_matrix[cm_label_idx, cm_fp_idx] = n_false_detections
          
      return confusion_matrix, confusion_matrix_labels
        
            
        
#     def show_spec(self, start_sec, end_sec, show_annotations = False, show_predictions = False, show_matched = False):
#         start_sample = int(self.sr * start_sec)
#         end_sample = int(self.sr *end_sec)
#         hop_length = 512
#         y = librosa.feature.melspectrogram(self.samples[start_sample:end_sample], sr = self.sr, hop_length = hop_length)
#         y = np.log(y + 1e-6)
#         # Create figure and axes
#         fig, ax = plt.subplots(figsize = (16, 6))
        
#         ticks = np.linspace(0, (end_sample - start_sample) // hop_length, num=10, dtype = int)
#         ticklabels = list(np.linspace(start_sec, end_sec, num=10, dtype = float))
#         ticklabels = ["{:1.1f}".format(x) for x in ticklabels]
#         ax.set_xticks(ticks)
#         ax.set_xticklabels(ticklabels)
#         ax.set_xlabel("Time (s)")
#         ax.imshow(y[::-1, :])
        
#         if show_annotations:
#             # Do we display annotations as boxes on the spectrogram?
            
#             # Only look at annotations that start within the window or end within the window
#             start_within_window = (self.annotations['Begin Time (s)'] > start_sec) & (self.annotations['Begin Time (s)'] < end_sec)
#             end_within_window = (self.annotations['End Time (s)'] > start_sec) & (self.annotations['End Time (s)'] < end_sec)
#             to_show = self.annotations[start_within_window | end_within_window]
            
#             jitter = 0
#             for i, row in to_show.iterrows():
#                 start_adjusted = row['Begin Time (s)'] - start_sec
#                 end_adjusted = row['End Time (s)'] - start_sec
#                 start_frame = int(start_adjusted * self.sr / hop_length)
#                 end_frame = int(end_adjusted * self.sr / hop_length)
                
#                 # Create a Rectangle patch
#                 if show_matched and row['index'] in self.matched_annotations:
#                     ec = (0,1,0,1)
#                 else:
#                     ec = (1,1,0,1)
                
#                 rect = patches.Rectangle((start_frame, 1 + jitter), end_frame - start_frame, 125, linewidth=1, edgecolor=ec, facecolor=(1,1,0,0.15))
#                 ax.add_patch(rect)
#                 jitter = (jitter + 1) % 2
                
#         if show_predictions:
#             # Do we display predictions as boxes on the spectrogram?
            
#             # Only look at predictions that start within the window or end within the window
#             start_within_window = (self.predictions['Begin Time (s)'] > start_sec) & (self.predictions['Begin Time (s)'] < end_sec)
#             end_within_window = (self.predictions['End Time (s)'] > start_sec) & (self.predictions['End Time (s)'] < end_sec)
#             to_show = self.predictions[start_within_window | end_within_window]
            
#             jitter = 0
#             for i, row in to_show.iterrows():
#                 start_adjusted = row['Begin Time (s)'] - start_sec
#                 end_adjusted = row['End Time (s)'] - start_sec
#                 start_frame = int(start_adjusted * self.sr / hop_length)
#                 end_frame = int(end_adjusted * self.sr / hop_length)
          
#                 # Create a Rectangle patch
#                 if show_matched and row['index'] in self.matched_predictions:
#                     ec = (0,1,0,1)
#                 else:
#                     ec = (1,0,0,1)
#                 rect = patches.Rectangle((start_frame, 1 + jitter), end_frame - start_frame, 125, linewidth=1, edgecolor=ec, facecolor=(1,0,0,0.15))
#                 ax.add_patch(rect)
#                 jitter = (jitter+1) % 2
        