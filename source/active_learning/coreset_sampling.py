import os
import numpy as np
import pandas as pd
import torch
import tqdm
import argparse

from source.training.params import load_params
from source.model.model import DetectionModel
from source.evaluation.evaluation import generate_features
from source.data.data import get_single_clip_data
from scipy.spatial import distance_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

def sample_coreset_all(al_args):
  if al_args.model_args_fp is not None:
    args = load_params(al_args.model_args_fp)
    model = DetectionModel(args)

    # model  
    model_checkpoint_fp = os.path.join(args.experiment_dir, "model.pt")
    print(f"Loading model weights from {model_checkpoint_fp}")
    cp = torch.load(model_checkpoint_fp)
    model.load_state_dict(cp["model_state_dict"])
  else:
    # If no detection model trained, use base weights for aves
    args = argparse.Namespace()
    for attr in ['sr', 'scale_factor', 'aves_model_weight_fp', 'rms_norm', 'clip_duration', 'batch_size', 'num_workers']:
      value = getattr(al_args, attr)
      setattr(args, attr, value)
      
    args.label_set =['placeholder']
    args.prediction_scale_factor = 1
    args.mixup = False
    
    model = DetectionModel(args)
  
  model = model.to(device)
  
  files_to_sample = pd.read_csv(al_args.train_pool_info_fp)
  
  # get previously sampled features
  if al_args.prev_iteration_info_fp is not None:
    print("Generating features from previously sampled audio")
    previously_sampled_fps = pd.read_csv(al_args.prev_iteration_info_fp)['audio_fp'].tolist()
    previous_features = []
    for fp in previously_sampled_fps:
      _, _, f = sample_features_fp(fp, model, al_args, args)
      previous_features.append(f)
    previous_features = np.concatenate(previous_features)
  else:
    previous_features = None
  
  # get features to sample
  start_times = []
  durations = []
  filenames = []
  filepaths = []
  features = []
  
  for i, row in files_to_sample.iterrows():
    fp = row['audio_fp']
    fn = row['fn']
    
    if not os.path.exists(fp):
      print(f"Could not locate file {fp}")
      continue
    print(f"Generating features from {fn}")
    
    s, d, f = sample_features_fp(fp, model, al_args, args)
    
    start_times.extend(s)
    durations.extend(d)
    filenames.extend([fn for i in s])
    filepaths.extend([fp for i in s])
    features.append(f)
    
  features = np.concatenate(features)
  print("Finished generating features")
  
  start_times_selected, durations_selected, filenames_selected, filepaths_selected = sample_coreset(start_times, durations, filenames, filepaths, features, previous_features, al_args)
  
    
  output_log = {'start_second' : start_times_selected,
                'duration' : durations_selected,
                'fn' : filenames_selected,
                'audio_fp' : filepaths_selected}
  output_log = pd.DataFrame(output_log)
  
  output_log = output_log.reset_index()
  output_log['start_second_in_al_samples'] = output_log.index * al_args.sample_duration
  
  return output_log
  
def sample_coreset(start_times, durations, filenames, filepaths, features, previous_features, al_args):
  
  # uses 2-Opt greedy solution
  n_samples = len(start_times)
  if n_samples < al_args.max_n_clips_to_sample:
    return start_times, durations, filenames, filepaths
  
  if previous_features is None:
    min_distance = np.full(n_samples, np.inf)
  else:
    d = distance_matrix(features, previous_features)
    min_distance = np.amin(d, axis = 1)
    
  start_times_selected = []
  durations_selected = []
  filenames_selected = []
  filepaths_selected = []
  
  print("Generating Coreset Samples")
  for i in tqdm.tqdm(range(al_args.max_n_clips_to_sample)):
    distance_to_farthest_point = np.amax(min_distance)
    farthest_point_idx = np.nonzero(min_distance == distance_to_farthest_point)[0][0]
    
    s = start_times[farthest_point_idx]
    d = durations[farthest_point_idx]
    fn = filenames[farthest_point_idx]
    fp = filepaths[farthest_point_idx]
    
    start_times_selected.append(s)
    durations_selected.append(d)
    filenames_selected.append(fn)
    filepaths_selected.append(fp)
    
    farthest_point_features = np.expand_dims(features[farthest_point_idx,:], 0)
    new_candidate_min_distance = distance_matrix(features, farthest_point_features)[:, 0]
    min_distance = np.minimum(min_distance, new_candidate_min_distance)
  
  return start_times_selected, durations_selected, filenames_selected, filepaths_selected

def sample_features_fp(audio_fp, model, al_args, args):  
  dataloader = get_single_clip_data(audio_fp, args.clip_duration/2, args)
  if len(dataloader) == 0:
    return [], [], []
  
  feats = generate_features(model, dataloader, args)
  feature_dimension = np.shape(feats)[-1]
  
  num_clips = int(np.floor((dataloader.dataset.duration - al_args.sample_duration) // al_args.sample_duration))
  
  # truncate before reshaping features into clips
  aves_sr = int(args.sr / (args.scale_factor * args.prediction_scale_factor))
  aves_samples_per_clip = int(aves_sr * al_args.sample_duration)
  feat_cutoff = num_clips * aves_samples_per_clip
  feats = feats[:feat_cutoff,:]
  
  starts = [al_args.sample_duration * i for i in range(num_clips)]
  durations = [al_args.sample_duration for i in range(num_clips)]
  feats = np.reshape(feats, (num_clips, -1, feature_dimension))
  feats = np.mean(feats, axis = 1) 
  return starts, durations, feats 