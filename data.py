import os
import math
import numpy as np
import pandas as pd
import librosa
import warnings

from numpy.random import default_rng
from intervaltree import IntervalTree
from torch.utils.data import Dataset, DataLoader

def normalize_sig_np(sig, eps=1e-8):
    sig = sig / (np.max(np.abs(sig))+eps)
    return sig

class DetectionDataset(Dataset):
    def __init__(self, info_df, clip_hop, train, args, random_seed_shift = 0):
        self.info_df = info_df
        self.label_set = args.label_set
        if hasattr(args, 'unknown_label'):
          self.unknown_label = args.unknown_label
        else:
          self.unknown_label = None
        self.label_mapping = args.label_mapping
        self.n_classes = len(self.label_set)
        self.sr = args.sr
        self.clip_duration = args.clip_duration
        self.clip_hop = clip_hop
        assert (self.clip_hop*args.sr).is_integer()
        self.seed = args.seed + random_seed_shift
        self.amp_aug = args.amp_aug
        if self.amp_aug:
          self.amp_aug_low_r = args.amp_aug_low_r
          self.amp_aug_high_r = args.amp_aug_high_r
          assert (self.amp_aug_low_r >= 0) #and (self.amp_aug_high_r <= 1) and 
          assert (self.amp_aug_low_r <= self.amp_aug_high_r)

        self.scale_factor = args.scale_factor
        self.prediction_scale_factor = args.prediction_scale_factor
        self.scale_factor_raw_to_prediction = self.scale_factor*self.prediction_scale_factor
        self.rng = default_rng(seed=self.seed)
        self.train=train
        
        if self.train:
          self.omit_empty_clip_prob = args.omit_empty_clip_prob
          self.clip_start_offset = self.rng.integers(0, np.floor(self.clip_hop*self.sr)) / self.sr
        else:
          self.omit_empty_clip_prob = 0
          self.clip_start_offset = 0
        
        # make metadata
        self.make_metadata()

    def augment_amplitude(self, signal):
        if not self.amp_aug:
            return signal
        else:
            r = self.rng.uniform(self.amp_aug_low_r, self.amp_aug_high_r)
            aug_signal = r*signal
            return aug_signal

    def process_timestamp(self, timestamp_fp):
        timestamp = pd.read_csv(timestamp_fp)
        tree = IntervalTree()

        for ii, row in timestamp.iterrows():
            start = row['start']
            end = row['end']
            label = row['label']
            if label in self.label_mapping:
              label = self.label_mapping[label]
            else:
              continue
            
            if label == self.unknown_label:
              label_idx = -1
            else:
              label_idx = self.label_set.index(label)
            tree.addi(start, end, label_idx)

        return tree

    def make_metadata(self):
        timestamp_dict = dict()
        metadata = []

        for ii, row in self.info_df.iterrows():
            fn = row['fn']
            duration = row['duration']
            audio_fp = row['audio_fp']
            timestamp_fp = row['timestamp_fp']

            timestamps = self.process_timestamp(timestamp_fp)

            timestamp_dict[fn] = timestamps

            num_clips = int(np.floor((duration - self.clip_duration - self.clip_start_offset) // self.clip_hop))

            for tt in range(num_clips):
                start = tt*self.clip_hop + self.clip_start_offset
                end = start + self.clip_duration

                ivs = timestamps[start:end]
                # if no annotated intervals, skip with specified probability
                if not ivs:
                  if self.omit_empty_clip_prob > self.rng.uniform():
                      continue

                metadata.append([fn, audio_fp, start, end])

                # self.get_sub_timestamps(timestamps)

        self.timestamp_dict = timestamp_dict
        self.metadata = metadata

    def get_pos_intervals(self, fn, start, end):
        tree = self.timestamp_dict[fn]

        intervals = tree[start:end]
        intervals = [(max(iv.begin, start)-start, min(iv.end, end)-start, iv.data) for iv in intervals]

        return intervals

    def get_annotation(self, pos_intervals, audio):
        
        raw_seq_len = audio.shape[0]
        seq_len = int(math.ceil(raw_seq_len / self.scale_factor_raw_to_prediction))
        regression_anno = np.zeros((seq_len, self.n_classes))

        anno_sr = int(self.sr // self.scale_factor_raw_to_prediction)
        
        anchor_annos_dict = {i : [] for i in range(self.n_classes)}
        
        mask = [np.ones(seq_len)]

        for iv in pos_intervals:
            start, end, class_idx = iv
            dur = end-start
            
            start_idx = int(math.floor(start*anno_sr))
            start_idx = max(min(start_idx, seq_len-1), 0)
            dur_samples = np.ceil(dur * anno_sr)
            
            # ### Start idx anchors and regression targets ###
            if class_idx != -1:
              anchor_anno = get_anchor_anno(start_idx, dur_samples, seq_len)
              anchor_annos_dict[class_idx].append(anchor_anno)
              regression_anno[start_idx, class_idx] = dur
            else:
              mask.append(get_unknown_mask(start_idx, dur_samples, seq_len))      
        
        anchor_annos = []
        for i in range(self.n_classes):
          if len(anchor_annos_dict[i])>0:
            x = np.stack(anchor_annos_dict[i])
            x = np.amax(x, axis = 0)
          else:
            x = np.zeros(seq_len)
          anchor_annos.append(x)
        
        anchor_annos = np.stack(anchor_annos, axis = -1)
          
        mask = np.stack(mask)
        mask = np.amin(mask, axis = 0)

        return anchor_annos, regression_anno, mask # shapes [time_steps, n_classes], [time_steps, n_classes], [time_steps,]

    def __getitem__(self, index):
        fn, audio_fp, start, end = self.metadata[index]
        audio, _ = librosa.load(audio_fp, sr=self.sr, offset=start, duration=self.clip_duration, mono=True)
        audio = audio-np.mean(audio)
        pos_intervals = self.get_pos_intervals(fn, start, end)
        anchor_anno, regression_anno, mask = self.get_annotation(pos_intervals, audio)

        if self.amp_aug and self.train:
            audio = self.augment_amplitude(audio)
                        
        return audio, anchor_anno, regression_anno, mask

    def __len__(self):
        return len(self.metadata)
      
      
def get_train_dataloader(args, random_seed_shift = 0):
  dev_info_fp = args.dev_info_fp
  dev_info_df = pd.read_csv(dev_info_fp)
  num_files_val = args.num_files_val
  if args.num_files_val>0:
    train_info_df = dev_info_df.iloc[:-args.num_files_val]
  else:
    train_info_df = dev_info_df
  
  train_dataset = DetectionDataset(train_info_df, args.clip_hop, True, args, random_seed_shift = random_seed_shift)
  train_dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size, 
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True, 
                                drop_last = True)
  
  return train_dataloader

def get_val_dataloader(args):
  dev_info_fp = args.dev_info_fp
  dev_info_df = pd.read_csv(dev_info_fp)
  num_files_val = args.num_files_val
  if args.num_files_val>0:
    val_info_df = dev_info_df.iloc[-args.num_files_val:]
  else:
    val_info_df = dev_info_df.iloc[-1:]
    warnings.warn("num_files_val was set to 0. I am using last dev file as validation, but this is also in the train set.")
  
  val_dataloaders = {}
  
  for i in range(len(val_info_df)):
    fn = val_info_df.iloc[i]['fn']
  
    val_file_dataset = DetectionDataset(val_info_df.iloc[i:i+1], args.clip_duration / 2, False, args)
    val_file_dataloader = DataLoader(val_file_dataset,
                                      batch_size=args.batch_size, 
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True, 
                                      drop_last = False)
    val_dataloaders[fn] = val_file_dataloader
    
  return val_dataloaders

def get_test_dataloader(args):
  test_info_fp = args.test_info_fp
  test_info_df = pd.read_csv(test_info_fp)
  
  test_dataloaders = {}
  
  for i in range(len(test_info_df)):
    fn = test_info_df.iloc[i]['fn']
  
    test_file_dataset = DetectionDataset(test_info_df.iloc[i:i+1], args.clip_duration / 2, False, args)
    test_file_dataloader = DataLoader(test_file_dataset,
                                      batch_size=args.batch_size, 
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True, 
                                      drop_last = False)
    test_dataloaders[fn] = test_file_dataloader
    
  
  return test_dataloaders
  
def get_anchor_anno(start_idx, dur_samples, seq_len):
  # start times plus gaussian blur
  std = dur_samples / 6
  x = (np.arange(seq_len) - start_idx) ** 2
  x = x / (2 * std**2)
  x = np.exp(-x)
  return x
  
def get_unknown_mask(start_idx, dur_samples, seq_len):
  unknown_radius = 1 + int(dur_samples / 6)
  x = np.ones(seq_len)
  x[start_idx - unknown_radius:start_idx + unknown_radius] = 0
  return x