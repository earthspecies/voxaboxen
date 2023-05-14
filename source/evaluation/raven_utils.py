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
import warnings

import source.evaluation.metrics as metrics

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
        if self.unknown_label is not None:
          confusion_matrix_labels.append(self.unknown_label)
        confusion_matrix_labels.append('None')
        confusion_matrix_size = len(confusion_matrix_labels)

        confusion_matrix = np.zeros((confusion_matrix_size, confusion_matrix_size))
        cm_nobox_idx = confusion_matrix_labels.index('None')
        
        pred_label = np.array(self.predictions['Annotation'])
        annot_label = np.array(self.annotations['Annotation'])
        
        for p in self.matching:
          annotation = annot_label[p[0]]
          prediction = pred_label[p[1]]
          cm_annot_idx = confusion_matrix_labels.index(annotation)
          cm_pred_idx = confusion_matrix_labels.index(prediction)
          confusion_matrix[cm_pred_idx, cm_annot_idx] += 1

        for label in confusion_matrix_labels:
          if label == 'None':
            continue
          # count false positive and false negative detections, regardless of class
          cm_label_idx = confusion_matrix_labels.index(label)
          
          #fp
          n_pred = int((pred_label == label).sum())
          n_positive_detections_row = confusion_matrix.sum(1)[cm_label_idx]
          n_false_detections = n_pred - n_positive_detections_row
          confusion_matrix[cm_label_idx, cm_nobox_idx] = n_false_detections
          
          #fn
          n_annot = int((annot_label == label).sum())
          n_positive_detections_col = confusion_matrix.sum(0)[cm_label_idx]
          n_missed_detections = n_annot - n_positive_detections_col
          confusion_matrix[cm_nobox_idx, cm_label_idx] = n_missed_detections
          
      return confusion_matrix, confusion_matrix_labels
        