# script to process data from https://zenodo.org/record/7050014
# assumes data are saved in raw_data_dir,
# with the exception of audio files which are saved in soundscape_data.
# Can use the tool zenodo_get to download zip files quickly

import pandas as pd
import os
import tqdm
import numpy as np
from glob import glob

def main():
  cwd = os.getcwd()
  
  raw_data_dir = os.path.join(cwd, 'raw')
  audio_dir = os.path.join(cwd, 'soundscape_data')
  
  formatted_data_dir = os.path.join(cwd, 'formatted')
  formatted_annot_dir = os.path.join(formatted_data_dir, 'selection_tables')
  for d in [formatted_data_dir, formatted_annot_dir]:
    if not os.path.exists(d):
      os.makedirs(d)
      
  raw_annotations_fp = os.path.join(cwd, 'raw', 'annotations.csv')
  raw_annot_df = pd.read_csv(raw_annotations_fp)
  raw_annot_df['Annotation'] = raw_annot_df['Species eBird Code']
  raw_annot_df = raw_annot_df.drop('Species eBird Code', axis=1)
  
  rng=np.random.default_rng(42)
  train_proportion = 0.5
  
  audio_fps = sorted(glob(os.path.join(audio_dir, "*.flac")))
  audio_fps = rng.permutation(audio_fps)
  
  n_train = int(train_proportion * len(audio_fps))
  
  train_audio_fps = audio_fps[:n_train]
  test_audio_fps = audio_fps[n_train:]
  
  train_fns = [os.path.basename(x) for x in train_audio_fps]
  test_fns = [os.path.basename(x) for x in test_audio_fps]
  
  train_annot_fps = []
  test_annot_fps = []
  
  for fn in train_fns:
    sub_annot_df = raw_annot_df[raw_annot_df['Filename'] == fn]
    sub_annot_df = sub_annot_df.drop('Filename', axis = 1)
    
    annot_fn = f"selection_table_{fn.split('.')[0]}.txt"
    annot_fp = os.path.join(formatted_annot_dir, annot_fn)
    
    sub_annot_df.to_csv(annot_fp, sep = '\t', index = False)
    train_annot_fps.append(annot_fp)
    
  for fn in test_fns:
    sub_annot_df = raw_annot_df[raw_annot_df['Filename'] == fn]
    sub_annot_df = sub_annot_df.drop('Filename', axis = 1)
    
    annot_fn = f"selection_table_{fn.split('.')[0]}.txt"
    annot_fp = os.path.join(formatted_annot_dir, annot_fn)
    
    sub_annot_df.to_csv(annot_fp, sep = '\t', index = False)
    test_annot_fps.append(annot_fp)
  
  train_info_df = pd.DataFrame({'fn' : train_fns, 'audio_fp' : train_audio_fps, 'selection_table_fp' : train_annot_fps})
  train_info_fp = os.path.join(formatted_data_dir, 'train_pool_info.csv')
  train_info_df.to_csv(train_info_fp, index = False)
  test_info_df = pd.DataFrame({'fn' : test_fns, 'audio_fp' : test_audio_fps, 'selection_table_fp' : test_annot_fps})
  test_info_fp = os.path.join(formatted_data_dir, 'test_info.csv')
  test_info_df.to_csv(test_info_fp, index = False)

if __name__ == "__main__":
  main()

