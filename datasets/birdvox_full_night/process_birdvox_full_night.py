# script to process data from https://zenodo.org/record/1205569
# assumes data are saved in raw_data_dir

import pandas as pd
import librosa
import os
import soundfile as sf
import tqdm
import numpy as np

def process_audio_and_annot(annot, audio, sr, train_proportion, voc_dur_sec = 0.2):
  audio_dur_samples = np.shape(audio)[0]
  audio_dur_sec = np.floor(audio_dur_samples / sr)
  train_audio_dur_samples = int(audio_dur_samples * train_proportion)
  train_audio_dur_sec = train_audio_dur_samples / sr
  
  train_audio = audio[:train_audio_dur_samples]
  test_audio = audio[train_audio_dur_samples:]
  
  voc_offset = voc_dur_sec / 2 # annotations are single points, at the middle of each call. We expand to bounding boxes.
  begin_time = list(np.maximum(0, np.array(annot['Time (s)'] - voc_offset)))
  end_time = list(np.minimum(audio_dur_sec, np.array(annot['Time (s)'] + voc_offset)))
  annotation = ['voc' for _ in begin_time]
  selection_table = pd.DataFrame({'Begin Time (s)' : begin_time, 'End Time (s)' : end_time, 'Annotation' : annotation})
  
  train_selection_table = selection_table[selection_table['End Time (s)'] < train_audio_dur_sec].copy()
  
  test_selection_table = selection_table[selection_table['Begin Time (s)'] >= train_audio_dur_sec].copy()
  test_selection_table['Begin Time (s)'] = test_selection_table['Begin Time (s)'] - train_audio_dur_sec
  test_selection_table['End Time (s)'] = test_selection_table['End Time (s)'] - train_audio_dur_sec
  
  return train_selection_table, train_audio, test_selection_table, test_audio

def main():  
  cwd = os.getcwd()
  
  raw_data_dir = os.path.join(cwd, 'raw')
  
  formatted_data_dir = os.path.join(cwd, 'formatted')
  formatted_audio_dir = os.path.join(formatted_data_dir, 'audio')
  formatted_annot_dir = os.path.join(formatted_data_dir, 'selection_tables')
  for d in [formatted_data_dir, formatted_audio_dir, formatted_annot_dir]:
    if not os.path.exists(d):
      os.makedirs(d)

  train_proportion = 0.5
  
  annotation_fns = ['BirdVox-full-night_csv-annotations_unit01.csv',
                    'BirdVox-full-night_csv-annotations_unit02.csv',
                    'BirdVox-full-night_csv-annotations_unit03.csv',
                    'BirdVox-full-night_csv-annotations_unit05.csv',
                    'BirdVox-full-night_csv-annotations_unit07.csv',
                    'BirdVox-full-night_csv-annotations_unit10.csv'
                   ]
  
  audio_fns = ['BirdVox-full-night_flac-audio_unit01.flac',
               'BirdVox-full-night_flac-audio_unit02.flac',
               'BirdVox-full-night_flac-audio_unit03.flac',
               'BirdVox-full-night_flac-audio_unit05.flac',
               'BirdVox-full-night_flac-audio_unit07.flac',
               'BirdVox-full-night_flac-audio_unit10.flac'
              ]
  
  train_fns = []
  train_audio_fps = []
  train_annot_fps = []
  
  test_fns = []
  test_audio_fps = []
  test_annot_fps = []
  
  for annot_fn, audio_fn in tqdm.tqdm(zip(annotation_fns, audio_fns)):
    fn = annot_fn.split('.')[0].split('_')[-1]
    train_fns.append(f"{fn}_train")
    test_fns.append(f"{fn}_test")
    
    annot_fp = os.path.join(raw_data_dir, annot_fn)
    audio_fp = os.path.join(raw_data_dir, audio_fn)
    
    annot = pd.read_csv(annot_fp)
    audio, sr = sf.read(audio_fp)
    
    train_selection_table, train_audio, test_selection_table, test_audio = process_audio_and_annot(annot, audio, sr, train_proportion)
  
    train_selection_table_fn = f"{annot_fn.split('.')[0]}_train.txt"
    train_selection_table_fp = os.path.join(formatted_annot_dir, train_selection_table_fn)
    train_selection_table.to_csv(train_selection_table_fp, sep = '\t', index = False)
    train_annot_fps.append(train_selection_table_fp)
    
    train_audio_fn = f"{audio_fn.split('.')[0]}_train.wav"
    train_audio_fp = os.path.join(formatted_audio_dir, train_audio_fn)
    sf.write(train_audio_fp, train_audio, sr)
    train_audio_fps.append(train_audio_fp)
    
    test_selection_table_fn = f"{annot_fn.split('.')[0]}_test.txt"
    test_selection_table_fp = os.path.join(formatted_annot_dir, test_selection_table_fn)
    test_selection_table.to_csv(test_selection_table_fp, sep = '\t', index = False)
    test_annot_fps.append(test_selection_table_fp)
    
    test_audio_fn = f"{audio_fn.split('.')[0]}_test.wav"
    test_audio_fp = os.path.join(formatted_audio_dir, test_audio_fn)
    sf.write(test_audio_fp, test_audio, sr)
    test_audio_fps.append(test_audio_fp)
  
  train_info_df = pd.DataFrame({'fn' : train_fns, 'audio_fp' : train_audio_fps, 'selection_table_fp' : train_annot_fps})
  train_info_fp = os.path.join(formatted_data_dir, 'train_pool_info.csv')
  train_info_df.to_csv(train_info_fp, index = False)
  test_info_df = pd.DataFrame({'fn' : test_fns, 'audio_fp' : test_audio_fps, 'selection_table_fp' : test_annot_fps})
  test_info_fp = os.path.join(formatted_data_dir, 'test_info.csv')
  test_info_df.to_csv(test_info_fp, index = False)

if __name__ == "__main__":
  main()
    