# python active_learning_sampling.py --model-args-fp=/home/jupyter/sound_event_detection/logs/four_classes_highlamb/params.yaml --model-checkpoint-fp=/home/jupyter/sound_event_detection/logs/four_classes_highlamb/final_model.pt --name=debug_al --clips-per-file=3 --candidate-manifest-fp=/home/jupyter/al_crows_debug.csv

import os
import yaml
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import librosa
import argparse
from scipy.signal import find_peaks
import soundfile as sf

from util import load_params, set_seed
from model import DetectionModel
from evaluation import generate_predictions
from data import get_single_clip_data

device = "cuda" if torch.cuda.is_available() else "cpu"

def sample_uncertainty(fp, model, al_args, args, rng):
  starts, durations, uncertainties = get_uncertainties(fp, model, al_args, args)
    
  uncertainty_cutoff = np.quantile(uncertainties, al_args.uncertainty_quantile)
  to_sample_idxs = np.nonzero(np.array(uncertainties) >= uncertainty_cutoff)[0]
  if len(to_sample_idxs)<al_args.clips_per_file:
    sampled_idxs = to_sample_idxs
  else:
    sampled_idxs = rng.choice(to_sample_idxs, size = al_args.clips_per_file, replace = False)
        
  audio_clips_sampled = [librosa.load(fp, offset = starts[i], duration = durations[i], sr=args.sr, mono=True)[0] for i in sampled_idxs]
  starts_sampled = [starts[i] for i in sampled_idxs]
  durations_sampled = [durations[i] for i in sampled_idxs]
  uncertainties_sampled = [uncertainties[i] for i in sampled_idxs]
  
  return audio_clips_sampled, starts_sampled, durations_sampled, uncertainties_sampled

def get_uncertainties(audio_fp, model, al_args, args):
  dataloader = get_single_clip_data(audio_fp, args.clip_duration/2, args)
  predictions, _ = generate_predictions(model, dataloader, args)
  
  if 'Morado' in audio_fp:
    np.save('/home/jupyter/test.npy', predictions)
  
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
        # uncertainties_masked = np.sort(uncertainties_masked)[-10:]
        uncertainties_masked = np.sort(uncertainties_masked)[::-1] 
        weighting = al_args.uncertainty_discount_factor ** np.arange(len(uncertainties_masked))
        uncertainties_masked = uncertainties_masked * weighting
        u += uncertainties_masked.sum()
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
  parser.add_argument('--uncertainty-quantile', type = float, default = .95, help = 'controls how clips are sampled, closer to 1 means more preference for the most uncertain clips')
  parser.add_argument('--clips-per-file', type = int, default = 1, help = 'how many clips to sample per file')
  parser.add_argument('--uncertainty-discount-factor', type = float, default = 0.8, help = 'geometric weighting of uncertainties. Uncertainties are sorted and then the lower ranking uncertainties count for less. Closer to 0 discourages sampling clips with lots of detections')
  parser.add_argument('--uncertainty-detection-threshold', type=float, default = 0.1, help = 'ignore detection peaks lower than this value, for the purpose of computing uncertainty')
  al_args = parser.parse_args()  
  return al_args
  
  
def main(al_args):
  al_args = parse_al_args(al_args)
  args = load_params(al_args.model_args_fp)
  set_seed(al_args.seed)
  rng = np.random.default_rng(al_args.seed)
  
  if al_args.output_dir is None:
    setattr(al_args, 'output_dir', args.experiment_dir)
  
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
  output_audio = []

  
  for i, row in files_to_sample.iterrows():
    fp = row['audio_fp']
    fn = row['fn']
    print(f"Sampling clips from {fn}")
    a, s, d, u = sample_uncertainty(fp, model, al_args, args, rng)
    output_audio.extend(a)
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
  output_log.to_csv(os.path.join(al_args.output_dir, f'{al_args.name}_active_learning_log.csv'))
  
  output_audio = np.concatenate(output_audio)
  output_audio_fp = os.path.join(al_args.output_dir, f'{al_args.name}_active_learning_audio.wav')
  
  print(f"Saving outputs to {al_args.output_dir}")
  sf.write(output_audio_fp, output_audio, args.sr)
  
if __name__ == "__main__":
    main(sys.argv[1:])
  