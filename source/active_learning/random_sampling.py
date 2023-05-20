import os
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
import librosa

def sample_random_all(al_args):
  
  if al_args.prev_iteration_info_fp is not None:
    raise ValueError('Random sampling is not implemented for multiple iterations')
  
  rng = np.random.default_rng(al_args.seed)
  
  files_to_sample = pd.read_csv(al_args.train_pool_info_fp)
    
  start_times = []
  durations = []
  filenames = []
  filepaths = []
  
  for i, row in files_to_sample.iterrows():
    fp = row['audio_fp']
    fn = row['fn']
    
    if not os.path.exists(fp):
      print(f"Could not locate file {fp}")
      continue
    print(f"Sampling clips from {fn}")
    
    s, d = sample_random_fp(fp, rng, al_args)
    
    start_times.extend(s)
    durations.extend(d)
    filenames.extend([fn for i in s])
    filepaths.extend([fp for i in s])
    
  output_log = {'start_second' : start_times,
                'duration' : durations,
                'fn' : filenames,
                'audio_fp' : filepaths,}
  
  output_log = pd.DataFrame(output_log)
  
  # subselect
  idxs_to_select = rng.permutation(np.arange(len(output_log)))[:al_args.max_n_clips_to_sample]
  output_log = output_log.iloc[idxs_to_select].reset_index()
  
  output_log['start_second_in_al_samples'] = output_log.index * al_args.sample_duration
  return output_log

def sample_random_fp(audio_fp, rng, al_args):
  if hasattr(al_args, "random_clips_per_file"):
    max_clips = al_args.random_clips_per_file
  else:
    max_clips = None
    
  duration = librosa.get_duration(filename=audio_fp)
    
  num_clips = int(np.floor((duration - al_args.sample_duration) // al_args.sample_duration))
  
  starts = np.arange(0, num_clips, al_args.sample_duration)
  
  if max_clips is not None:
    starts = rng.permutation(starts)[:max_clips]
    
  starts = list(starts)
  durations = [al_args.sample_duration for _ in starts]
  
  return starts, durations
  