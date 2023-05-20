import os
import sys
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
import soundfile as sf

from source.training.params import set_seed
from source.active_learning.params import parse_al_args, save_al_params, expand_al_args
from source.data.data import crop_and_pad
from source.active_learning.uncertainty_sampling import sample_uncertainty_all
from source.active_learning.coreset_sampling import sample_coreset_all
from source.active_learning.random_sampling import sample_random_all
from source.active_learning.query_oracle import query_oracle
    
def assemble_output_audio(output_log, al_args):
  output_audio = []
  for i, row in output_log.iterrows():
    fp = row['audio_fp']
    start = row['start_second']
    duration = row['duration']
    
    audio, file_sr = librosa.load(fp, offset = start, duration = duration, sr = None, mono= True)
    audio = audio-np.mean(audio)
    audio = torch.from_numpy(audio)
    
    if file_sr != al_args.output_sr:
      audio = torchaudio.functional.resample(audio, file_sr, al_args.output_sr) 
      audio = crop_and_pad(audio, al_args.output_sr, duration)
      
    audio = audio.numpy()
    output_audio.append(audio)
  
  output_audio = np.concatenate(output_audio)
  return output_audio
  
def active_learning_sampling(al_args):
  al_args = parse_al_args(al_args)
  set_seed(al_args.seed)
  
  al_args = expand_al_args(al_args)
  save_al_params(al_args)
  
  if al_args.sampling_method == 'uncertainty':
    output_log = sample_uncertainty_all(al_args)
    
  elif al_args.sampling_method == 'random':
    output_log = sample_random_all(al_args)
    
  elif al_args.sampling_method == 'coreset':
    output_log = sample_coreset_all(al_args)
    
  else:
    raise NotImplementedError
  
  output_log.to_csv(os.path.join(al_args.output_dir, f'{al_args.name}_active_learning_log.csv'))
  output_audio = assemble_output_audio(output_log, al_args)
  output_audio_fp = os.path.join(al_args.output_dir, f'{al_args.name}_active_learning_audio.wav')
  
  print(f"Saving outputs to {al_args.output_dir}")
  sf.write(output_audio_fp, output_audio, al_args.output_sr)
  
  # query oracle
  if al_args.query_oracle:
    selection_table_fp = query_oracle(al_args, output_log)
  else:
    selection_table_fp = "None"
  
  # add to previous train info
  train_info = {'fn' : [f"{al_args.name}_active_learning_audio"], 'audio_fp' : [output_audio_fp], 'selection_table_fp' : [selection_table_fp]}
  train_info = pd.DataFrame(train_info)
  
  if al_args.prev_iteration_info_fp is not None:
    old_train_info = pd.read_csv(al_args.prev_iteration_info_fp)
    train_info = pd.concat([old_train_info, train_info])
  
  train_info_fp = os.path.join(al_args.output_dir, f"train_info_{al_args.name}.csv")
  train_info.to_csv(train_info_fp, index=False)
  
  
if __name__ == "__main__":
    main(sys.argv[1:])
  