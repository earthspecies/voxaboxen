import os
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks

from source.training.params import load_params
from source.model.model import DetectionModel
from source.evaluation.evaluation import generate_predictions
from source.data.data import get_single_clip_data

device = "cuda" if torch.cuda.is_available() else "cpu"

def sample_uncertainty_all(al_args):
  args = load_params(al_args.model_args_fp)
  
    
  project_config = load_params(al_args.project_config_fp)
  
  model = DetectionModel(args)
  
  # model  
  model_checkpoint_fp = os.path.join(args.experiment_dir, "model.pt")
  print(f"Loading model weights from {model_checkpoint_fp}")
  cp = torch.load(model_checkpoint_fp)
  model.load_state_dict(cp["model_state_dict"])
  model = model.to(device)
  
  files_to_sample = pd.read_csv(project_config.train_pool_info_fp)
  
  start_times = []
  durations = []
  filenames = []
  filepaths = []
  uncertainties = []
  
  for i, row in files_to_sample.iterrows():
    fp = row['audio_fp']
    fn = row['fn']
    
    if not os.path.exists(fp):
      print(f"Could not locate file {fp}")
      continue
    print(f"Sampling clips from {fn}")
    
    s, d, u = sample_uncertainty_fp(fp, model, al_args, args)
    
    start_times.extend(s)
    durations.extend(d)
    filenames.extend([fn for i in s])
    filepaths.extend([fp for i in s])
    uncertainties.extend(u)
    
  output_log = {'start_second' : start_times,
                'duration' : durations,
                'fn' : filenames,
                'audio_fp' : filepaths,
                'uncertainty' : uncertainties}
  output_log = pd.DataFrame(output_log)
  
  # subselect
  output_log = output_log.sort_values('uncertainty', ascending = False)[:al_args.max_n_clips_to_sample].reset_index()
  
  return output_log

def sample_uncertainty_fp(fp, model, al_args, args):
  starts, durations, uncertainties = get_uncertainties(fp, model, al_args, args)
  
  if len(uncertainties)==0:
    return [], [], []
  
  if len(uncertainties)<=al_args.uncertainty_clips_per_file:
    return starts, durations, uncertainties
  
  unc_sorted = np.sort(uncertainties)[::-1]
  uncertainty_cutoff = unc_sorted[al_args.uncertainty_clips_per_file-1]
  
  # sample above cutoff
  sampled_idxs = np.nonzero(np.array(uncertainties) > uncertainty_cutoff)[0]
  
  # sample at cutoff, handle situation if there are ties in uncertainty
  n_remaining_to_sample = al_args.uncertainty_clips_per_file - len(sampled_idxs)
  borderline_sampled_idxs = np.nonzero(np.array(uncertainties) == uncertainty_cutoff)[0]
  borderline_sampled_idxs = borderline_sampled_idxs[:n_remaining_to_sample]
  
  sampled_idxs = np.concatenate([sampled_idxs, borderline_sampled_idxs])
        
  starts_sampled = [starts[i] for i in sampled_idxs]
  durations_sampled = [durations[i] for i in sampled_idxs]
  uncertainties_sampled = [uncertainties[i] for i in sampled_idxs]
  
  return starts_sampled, durations_sampled, uncertainties_sampled

def get_uncertainties(audio_fp, model, al_args, args):
  dataloader = get_single_clip_data(audio_fp, args.clip_duration/2, args)
  if len(dataloader) == 0:
    return [], [], []
  
  predictions, _ = generate_predictions(model, dataloader, args)
  
  num_clips = int(np.floor((dataloader.dataset.duration - al_args.sample_duration) // al_args.sample_duration))
  
  starts = [al_args.sample_duration * i for i in range(num_clips)]
  durations = [al_args.sample_duration for i in range(num_clips)]
  uncertainties = compute_batched_uncertainty(predictions, starts, durations, args, al_args)
 
  return starts, durations, uncertainties

def compute_batched_uncertainty(predictions, starts, durations, args, al_args):
  peaks = {}
  uncertainties_dict = {}
  n_classes = np.shape(predictions)[-1]
  model_prediction_sr = int(args.sr / (args.scale_factor * args.prediction_scale_factor))
  
  for i in range(n_classes):
    x = predictions[:,i]
    p, d = find_peaks(x, height=al_args.uncertainty_detection_threshold, distance=5)
    peaks[i] = p / model_prediction_sr # location of peaks in seconds
    heights = d['peak_heights']
    uncertainties_dict[i] = 1-np.abs(1-2*heights)
    
  uncertainties = []
  for start, duration in zip(starts, durations):
    u = 0
    for i in range(n_classes):
      if len(uncertainties_dict[i])>0:
        mask = ((peaks[i] >= start) * (peaks[i] < (start+duration))).astype(float)
        uncertainties_masked = uncertainties_dict[i] * mask
        uncertainties_masked = np.sort(uncertainties_masked)[::-1] 
        weighting = al_args.uncertainty_discount_factor ** np.arange(len(uncertainties_masked))
        uncertainties_weighted = uncertainties_masked * weighting
        u += uncertainties_weighted.sum()
    uncertainties.append(float(u))
    
  return uncertainties