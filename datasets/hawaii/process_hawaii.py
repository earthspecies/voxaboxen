# script to process data from https://zenodo.org/record/7079380
# assumes raw selection tables are saved in raw/selection_tables,
# with the exception of audio files which are saved in soundscape_data.
# Can use the tool zenodo_get to download zip files quickly
# If encountering issues with loading flac with librosa, can convert to .wav with: for i in *.flac; do ffmpeg -i "$i" "${i%.*}.wav"; done

import pandas as pd
import os
import tqdm
import numpy as np
from glob import glob

file_extension = 'wav' # use 'flac' if not converted

def main():
  cwd = os.getcwd()

  raw_data_dir = os.path.join(cwd, 'raw')
  audio_dir = os.path.join(cwd, 'soundscape_data')

  raw_annotations_fp = os.path.join(cwd, 'raw', 'annotations.csv')
  raw_annot_df = pd.read_csv(raw_annotations_fp)
  raw_annot_df['Annotation'] = raw_annot_df['Species eBird Code']
  raw_annot_df = raw_annot_df.drop('Species eBird Code', axis=1)

  raw_annot_df['Begin Time (s)'] = raw_annot_df['Start Time (s)']
  raw_annot_df = raw_annot_df.drop('Start Time (s)', axis=1)

  formatted_data_dir = os.path.join(cwd, 'formatted')
  formatted_annot_dir = os.path.join(formatted_data_dir, 'selection_tables')
  for d in [formatted_data_dir, formatted_annot_dir]:
    if not os.path.exists(d):
      os.makedirs(d)

  train_proportion = 0.6
  val_proportion = 0.2

  train_audio_fps = []
  val_audio_fps = []
  test_audio_fps = []

  for i in range(1,5):
    audio_fps = sorted(glob(os.path.join(audio_dir, f"*_S0{i}_*.{file_extension}")))
    #audio_fps = sorted(glob(os.path.join(audio_dir, f"*Recording_{i}_*.{file_extension}")))
    n_train = int(train_proportion * len(audio_fps))
    n_val = int(val_proportion * len(audio_fps))

    train_audio_fps.extend(audio_fps[:n_train])
    val_audio_fps.extend(audio_fps[n_train:n_train+n_val])
    test_audio_fps.extend(audio_fps[n_train+n_val:])

  train_fns = [os.path.basename(x).split('.')[0] for x in train_audio_fps]
  val_fns = [os.path.basename(x).split('.')[0] for x in val_audio_fps]
  test_fns = [os.path.basename(x).split('.')[0] for x in test_audio_fps]

  train_annot_fps = []
  val_annot_fps = []
  test_annot_fps = []

  for fn, audio_fp in zip(train_fns, train_audio_fps):
    sub_annot_df = raw_annot_df[raw_annot_df['Filename'] == f'{fn}.flac']
    sub_annot_df = sub_annot_df.drop('Filename', axis = 1)

    annot_fn = f"selection_table_{fn.split('.')[0]}.txt"
    annot_fp = os.path.join(formatted_annot_dir, annot_fn)

    sub_annot_df.to_csv(annot_fp, sep = '\t', index = False)
    train_annot_fps.append(annot_fp)

  for fn, audio_fp in zip(val_fns, val_audio_fps):
    sub_annot_df = raw_annot_df[raw_annot_df['Filename'] == f'{fn}.flac']
    sub_annot_df = sub_annot_df.drop('Filename', axis = 1)

    annot_fn = f"selection_table_{fn.split('.')[0]}.txt"
    annot_fp = os.path.join(formatted_annot_dir, annot_fn)

    sub_annot_df.to_csv(annot_fp, sep = '\t', index = False)
    val_annot_fps.append(annot_fp)

  for fn, audio_fp in zip(test_fns, test_audio_fps):
    sub_annot_df = raw_annot_df[raw_annot_df['Filename'] == f'{fn}.flac']
    sub_annot_df = sub_annot_df.drop('Filename', axis = 1)

    annot_fn = f"selection_table_{fn.split('.')[0]}.txt"
    annot_fp = os.path.join(formatted_annot_dir, annot_fn)

    sub_annot_df.to_csv(annot_fp, sep = '\t', index = False)
    test_annot_fps.append(annot_fp)

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
