# python active_learning_sampling.py --model-args-fp=/home/jupyter/sound_event_detection/logs/sweep3/params.yaml --model-checkpoint-fp=/home/jupyter/sound_event_detection/logs/sweep3/best_model.pt --name=active_learning_19_iter0 --clips-per-file=1 --candidate-manifest-fp=/home/jupyter/carrion_crows_data/call_detection_data.active_origsr/unannotated_info.csv --output-dir=/home/jupyter/sound_event_detection/active_learning_19_iter0

import os
import yaml
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
import librosa
import argparse
from scipy.signal import find_peaks
import soundfile as sf

from util import load_params, set_seed
from model import DetectionModel
from evaluation import generate_predictions
from data import get_single_clip_data, crop_and_pad

device = "cuda" if torch.cuda.is_available() else "cpu"

def sample_uncertainty(fp, model, al_args, args):
  starts, durations, uncertainties = get_uncertainties(fp, model, al_args, args)
  
  if len(uncertainties)==0:
    return [], [], []
  
  if len(uncertainties)<=al_args.clips_per_file:
    return starts, durations, uncertainties
  
  unc_sorted = np.sort(uncertainties)[::-1]
  uncertainty_cutoff = unc_sorted[al_args.clips_per_file-1]
  
  # sample above cutoff
  sampled_idxs = np.nonzero(np.array(uncertainties) > uncertainty_cutoff)[0]
  
  # sample at cutoff, handle situation if there are ties in uncertainty
  n_remaining_to_sample = al_args.clips_per_file - len(sampled_idxs)
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
  
def parse_al_args(al_args):
  parser = argparse.ArgumentParser()
  
  # General
  parser.add_argument('--model-args-fp', type=str, required=True, help = "filepath of model params saved as a yaml")
  parser.add_argument('--model-checkpoint-fp', type=str, required=True, help = "filepath of model checkpoint")
  parser.add_argument('--seed', type=int, default=0, help="random seed")
  parser.add_argument('--name', type=str, required=True)
  parser.add_argument('--output-dir', type=str, default=None)
  parser.add_argument('--candidate-manifest-fp', help='fp to tsv list of wav files to sample')
  parser.add_argument('--sample-duration', type = float, default = 30, help = 'duration of clip to sample for annotation, in seconds')
  parser.add_argument('--max-n-clips-to-sample', type = int, default = 120, help = 'how many clips to sample')
  parser.add_argument('--clips-per-file', type = int, default = 1, help = 'how many clips to sample per file')
  parser.add_argument('--uncertainty-discount-factor', type = float, default = 0.8, help = 'geometric weighting of uncertainties. Uncertainties are sorted and then the lower ranking uncertainties count for less. Closer to 0 discourages sampling clips with lots of detections')
  parser.add_argument('--uncertainty-detection-threshold', type=float, default = 0.1, help = 'ignore detection peaks lower than this value, for the purpose of computing uncertainty')
  al_args = parser.parse_args()  
  return al_args

def save_params(al_args):
  """ Save a copy of the params used for this experiment """
  params_file = os.path.join(al_args.output_dir, f"{al_args.name}_params.yaml")

  args_dict = {}
  for name in sorted(vars(al_args)):
    val = getattr(al_args, name)
    args_dict[name] = val

  with open(params_file, "w") as f:
    yaml.dump(args_dict, f)
    
def assemble_output_audio(output_log, al_args, args):
  output_audio = []
  for i, row in output_log.iterrows():
    fp = row['audio_fp']
    start = row['start_second']
    duration = row['duration']
    
    audio, file_sr = librosa.load(fp, offset = start, duration = duration, sr = None, mono= True)
    audio = audio-np.mean(audio)
    audio = torch.from_numpy(audio)
    
    if file_sr != args.sr:
      audio = torchaudio.functional.resample(audio, file_sr, args.sr) 
      audio = crop_and_pad(audio, args.sr, duration)
      
    audio = audio.numpy()
    output_audio.append(audio)
  
  output_audio = np.concatenate(output_audio)
  return output_audio
  
def main(al_args):
  al_args = parse_al_args(al_args)
  args = load_params(al_args.model_args_fp)
  set_seed(al_args.seed)
  
  if al_args.output_dir is None:
    setattr(al_args, 'output_dir', args.experiment_dir)
    
  if not os.path.exists(al_args.output_dir):
    os.makedirs(al_args.output_dir)
    
  save_params(al_args)
  
  model = DetectionModel(args)
  
  # model  
  print(f"Loading model weights from {al_args.model_checkpoint_fp}")
  cp = torch.load(al_args.model_checkpoint_fp)
  model.load_state_dict(cp["model_state_dict"])
  model = model.to(device)
  
  files_to_sample = pd.read_csv(al_args.candidate_manifest_fp)
  
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
    
    s, d, u = sample_uncertainty(fp, model, al_args, args)
    
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
  
  output_log.to_csv(os.path.join(al_args.output_dir, f'{al_args.name}_active_learning_log.csv'))
  output_audio = assemble_output_audio(output_log, al_args, args)
  output_audio_fp = os.path.join(al_args.output_dir, f'{al_args.name}_active_learning_audio.wav')
  
  print(f"Saving outputs to {al_args.output_dir}")
  sf.write(output_audio_fp, output_audio, args.sr)
  
if __name__ == "__main__":
    main(sys.argv[1:])
  