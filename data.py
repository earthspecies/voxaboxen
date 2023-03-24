import os
import math
import numpy as np
import pandas as pd
import librosa

from numpy.random import RandomState
from intervaltree import IntervalTree
from torch.utils.data import Dataset, DataLoader

def normalize_sig_np(sig, eps=1e-8):
    sig = sig / (np.max(np.abs(sig))+eps)
    return sig


class DetectionDataset(Dataset):
    def __init__(self, info_df, clip_hop, args):
        self.info_df = info_df
        self.anchor_win_sizes = np.array(args.anchor_durs_sec)        
        self.label_set = args.label_set
        self.sr = args.sr
        self.clip_duration = args.clip_duration
        self.clip_hop = clip_hop
        self.seed = args.seed
        self.amp_aug = args.amp_aug
        if self.amp_aug:
          self.amp_aug_low_r = args.amp_aug_low_r
          self.amp_aug_high_r = args.amp_aug_high_r
          assert (self.amp_aug_low_r >= 0) and (self.amp_aug_high_r <= 1) and (self.amp_aug_low_r <= self.amp_aug_high_r)

        self.scale_factor = args.scale_factor
        self.rs = RandomState(seed=self.seed)
        
        # make metadata
        self.make_metadata()

    def augment_amplitude(self, signal):
        if not self.amp_aug:
            return signal
        else:
            r = self.rs.uniform(self.amp_aug_low_r, self.amp_aug_high_r)
            aug_signal = r*signal
            return aug_signal

    def process_timestamp(self, timestamp_fp):
        timestamp = pd.read_csv(timestamp_fp)
        tree = IntervalTree()

        for ii, row in timestamp.iterrows():
            start = row['start']
            end = row['end']
            label_idx = self.label_set.index(row['label'])

            tree.addi(start, end, label_idx)
            # print(start, end, label_idx)

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

            num_clips = int(math.floor(duration - self.clip_duration) / self.clip_hop)

            for tt in range(num_clips):
                start = tt*self.clip_hop
                end = start + self.clip_duration

                ivs = timestamps[start:end]
                # if no positive intervals, skip
                if not ivs:
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

    def get_annotation(self, anchor_win_sizes, pos_intervals, audio):
        raw_seq_len = audio.shape[0]

        num_anchors = len(anchor_win_sizes)

        seq_len = int(math.ceil(raw_seq_len / self.scale_factor))

        anchor_anno = np.zeros((seq_len, num_anchors), dtype=np.int64)
        class_anno = np.zeros(seq_len, dtype=np.int64) - 1

        anno_sr = int(self.sr // self.scale_factor)

        for iv in pos_intervals:
            start, end, class_idx = iv
            # start_idx = int(math.ceil(start*anno_sr))
            # end_idx = int(math.floor(end*anno_sr))

            center = (start+end) / 2
            center_idx = int(round(center*anno_sr))
            center_idx = max(min(center_idx, seq_len-1), 0)

            # ### Anchors ###
            interval_win_size = end-start
            anchor_idx = np.argmin(np.abs(self.anchor_win_sizes-interval_win_size))
            anchor_anno[center_idx, anchor_idx] = 1

            # ### Class ###
            class_anno[center_idx] = class_idx

        return anchor_anno, class_anno

    def __getitem__(self, index):
        fn, audio_fp, start, end = self.metadata[index]
        audio, _ = librosa.load(audio_fp, sr=self.sr, offset=start, duration=end-start, mono=True)
        pos_intervals = self.get_pos_intervals(fn, start, end)
        anchor_anno, class_anno = self.get_annotation(self.anchor_win_sizes, pos_intervals, audio)

        # ### Data Aug: amplitude ###
        if self.amp_aug:
            audio = normalize_sig_np(audio)
            audio = self.augment_amplitude(audio)

        return audio, anchor_anno, class_anno

    def __len__(self):
        return len(self.metadata)
      
      
def get_dataloader(args):
  dev_info_fp = args.dev_info_fp
  dev_info_df = pd.read_csv(dev_info_fp)
  num_files_val = args.num_files_val
  train_info_df = dev_info_df.iloc[:-args.num_files_val]
  val_info_df = dev_info_df.iloc[-args.num_files_val:]
  test_info_fp = args.test_info_fp
  test_info_df = pd.read_csv(test_info_fp)
  
  train_dataset = DetectionDataset(train_info_df, args.clip_hop, args)
  train_dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size, 
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True, 
                                drop_last = True)
  
  val_dataset = DetectionDataset(val_info_df, args.clip_duration, args)
  val_dataloader = DataLoader(val_dataset,
                              batch_size=args.batch_size, 
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=True, 
                              drop_last = False)
  
  test_dataloaders = {}
  
  for i in range(len(test_info_df)):
    fn = test_info_df.iloc[i]['fn']
  
    test_file_dataset = DetectionDataset(test_info_df.iloc[i:i+1], args.clip_duration, args)
    test_file_dataloader = DataLoader(test_file_dataset,
                                      batch_size=args.batch_size, 
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True, 
                                      drop_last = False)
    test_dataloaders[fn] = test_file_dataloader
    
  
  return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloaders}
  
  
  

# Example, old use:
      


# if __name__ == "__main__":

#     anchor_win_sizes = [0.1, 0.3, 0.5]

#     label_set = [
#         'crow',
#     ]

#     amp_aug_config = {
#         'r_range': [0.1, 1],
#         'seed': 123
#     }

#     sr = 16000
#     clip_duration = 30
#     clip_hop = 15

#     scale_factor = 320

#     info_dir = '/home/jupyter/storage/Datasets/spanish_carrion_crows/call_detection_data.revised_anno.all_crows/'

#     info_fp = os.path.join(info_dir, 'dev_info.csv')
#     # info_fp = os.path.join(info_dir, 'test_info.csv')
#     info = pd.read_csv(info_fp)
#     num_files_va = 1
#     info_tr = info.iloc[:-num_files_va]
#     info_va = info.iloc[-num_files_va:]

#     dataset = DetectionDataset(anchor_win_sizes, info_va, label_set, sr, clip_duration, clip_hop, scale_factor, amp_aug_config=amp_aug_config)

#     item = dataset.__getitem__(0)

#     # for ii in range(100):
#     #     item = dataset.__getitem__(ii)
