# script to process MT data from https://zenodo.org/record/5412896
# assumes data are saved in raw_data_dir

import pandas as pd
import librosa
import os
import soundfile as sf
import tqdm
import numpy as np
from glob import glob

def process_audio_and_annot(annot, audio, sr, train_proportion, val_proportion, labels = ['SNMK', 'CCMK', 'AGGM', 'SOCM']):
  low_hz = 100
  high_hz = sr/2 - low_hz
  audio_dur_samples = np.shape(audio)[0]
  audio_dur_sec = np.floor(audio_dur_samples / sr)
  train_audio_dur_samples = int(audio_dur_samples * train_proportion)
  train_audio_dur_sec = train_audio_dur_samples / sr
  
  val_audio_dur_samples = int(audio_dur_samples * val_proportion)
  val_audio_dur_sec = val_audio_dur_samples / sr
  
  train_audio = audio[:train_audio_dur_samples]
  val_audio = audio[train_audio_dur_samples:train_audio_dur_samples+val_audio_dur_samples]
  test_audio = audio[train_audio_dur_samples+val_audio_dur_samples:]
  
  ys = []
  for i, row in annot.iterrows():
    y = 'voc'
    # y = 'Unknown'
    # for label in labels:
    #   if row[label] == 'POS':
    #     y = label
    ys.append(y)
    
  begin_time = list(annot['Starttime'])
  end_time = list(annot['Endtime'])
  
  selection_table = pd.DataFrame({'Begin Time (s)' : begin_time, 'End Time (s)' : end_time, 'Annotation' : ys, 'Low Freq (Hz)' : [low_hz for x in begin_time], 'High Freq (Hz)' : [high_hz for x in begin_time]}).drop_duplicates()
  
  train_selection_table = selection_table[selection_table['End Time (s)'] < train_audio_dur_sec].copy()
  val_selection_table = selection_table[(selection_table['End Time (s)'] >= train_audio_dur_sec) & (selection_table['End Time (s)'] < train_audio_dur_sec + val_audio_dur_sec)].copy()
  val_selection_table['Begin Time (s)'] = val_selection_table['Begin Time (s)'] - train_audio_dur_sec
  val_selection_table['Begin Time (s)'] = val_selection_table['Begin Time (s)'].map(lambda x : max(x, 0))
  val_selection_table['End Time (s)'] = val_selection_table['End Time (s)'] - train_audio_dur_sec
  
  test_selection_table = selection_table[selection_table['Begin Time (s)'] >= train_audio_dur_sec + val_audio_dur_sec].copy()
  test_selection_table['Begin Time (s)'] = test_selection_table['Begin Time (s)'] - (train_audio_dur_sec + val_audio_dur_sec)
  test_selection_table['Begin Time (s)'] = test_selection_table['Begin Time (s)'].map(lambda x : max(x, 0))
  test_selection_table['End Time (s)'] = test_selection_table['End Time (s)'] - (train_audio_dur_sec + val_audio_dur_sec)
  
  return train_selection_table, train_audio, val_selection_table, val_audio, test_selection_table, test_audio

def main():  
  cwd = os.getcwd()
  
  raw_data_dir = os.path.join(cwd, 'raw')
  
  formatted_data_dir = os.path.join(cwd, 'formatted')
  formatted_audio_dir = os.path.join(formatted_data_dir, 'audio')
  formatted_annot_dir = os.path.join(formatted_data_dir, 'selection_tables')
  for d in [formatted_data_dir, formatted_audio_dir, formatted_annot_dir]:
    if not os.path.exists(d):
      os.makedirs(d)

  train_proportion = 0.6
  val_proportion = 0.2
  
  annotation_fns = sorted(glob(os.path.join(raw_data_dir, '*.csv')))
  annotation_fns = [os.path.basename(x) for x in annotation_fns]
  audio_fns = sorted(glob(os.path.join(raw_data_dir, '*.wav')))
  audio_fns = [os.path.basename(x) for x in audio_fns]
  
  train_fns = []
  train_audio_fps = []
  train_annot_fps = []
  
  val_fns = []
  val_audio_fps = []
  val_annot_fps = []
  
  test_fns = []
  test_audio_fps = []
  test_annot_fps = []
  
  for annot_fn, audio_fn in tqdm.tqdm(zip(annotation_fns, audio_fns)):
    fn = annot_fn.split('.')[0]
    train_fns.append(f"{fn}_train")
    val_fns.append(f"{fn}_val")
    test_fns.append(f"{fn}_test")
    
    annot_fp = os.path.join(raw_data_dir, annot_fn)
    audio_fp = os.path.join(raw_data_dir, audio_fn)
    
    annot = pd.read_csv(annot_fp)
    audio, sr = sf.read(audio_fp)
    
    train_selection_table, train_audio, val_selection_table, val_audio, test_selection_table, test_audio = process_audio_and_annot(annot, audio, sr, train_proportion, val_proportion)
  
    train_selection_table_fn = f"{annot_fn.split('.')[0]}_train.txt"
    train_selection_table_fp = os.path.join(formatted_annot_dir, train_selection_table_fn)
    train_selection_table.to_csv(train_selection_table_fp, sep = '\t', index = False)
    train_annot_fps.append(train_selection_table_fp)
    
    train_audio_fn = f"{audio_fn.split('.')[0]}_train.wav"
    train_audio_fp = os.path.join(formatted_audio_dir, train_audio_fn)
    sf.write(train_audio_fp, train_audio, sr)
    train_audio_fps.append(train_audio_fp)
    
    val_selection_table_fn = f"{annot_fn.split('.')[0]}_val.txt"
    val_selection_table_fp = os.path.join(formatted_annot_dir, val_selection_table_fn)
    val_selection_table.to_csv(val_selection_table_fp, sep = '\t', index = False)
    val_annot_fps.append(val_selection_table_fp)
    
    val_audio_fn = f"{audio_fn.split('.')[0]}_val.wav"
    val_audio_fp = os.path.join(formatted_audio_dir, val_audio_fn)
    sf.write(val_audio_fp, val_audio, sr)
    val_audio_fps.append(val_audio_fp)
    
    test_selection_table_fn = f"{annot_fn.split('.')[0]}_test.txt"
    test_selection_table_fp = os.path.join(formatted_annot_dir, test_selection_table_fn)
    test_selection_table.to_csv(test_selection_table_fp, sep = '\t', index = False)
    test_annot_fps.append(test_selection_table_fp)
    
    test_audio_fn = f"{audio_fn.split('.')[0]}_test.wav"
    test_audio_fp = os.path.join(formatted_audio_dir, test_audio_fn)
    sf.write(test_audio_fp, test_audio, sr)
    test_audio_fps.append(test_audio_fp)

    # Clear variables to free memory
    audio = None
    test_audio = None
    val_audio = None
    train_audio = None
  
  train_info_df = pd.DataFrame({'fn' : train_fns, 'audio_fp' : train_audio_fps, 'selection_table_fp' : train_annot_fps})
  train_info_fp = os.path.join(formatted_data_dir, 'train_info.csv')
  train_info_df.to_csv(train_info_fp, index = False)
  val_info_df = pd.DataFrame({'fn' : val_fns, 'audio_fp' : val_audio_fps, 'selection_table_fp' : val_annot_fps})
  val_info_fp = os.path.join(formatted_data_dir, 'val_info.csv')
  val_info_df.to_csv(val_info_fp, index = False)
  test_info_df = pd.DataFrame({'fn' : test_fns, 'audio_fp' : test_audio_fps, 'selection_table_fp' : test_annot_fps})
  test_info_fp = os.path.join(formatted_data_dir, 'test_info.csv')
  test_info_df.to_csv(test_info_fp, index = False)

if __name__ == "__main__":
  main()

