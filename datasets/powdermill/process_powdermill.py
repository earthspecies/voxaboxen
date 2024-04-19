# script to process data from https://datadryad.org/stash/dataset/doi:10.5061/dryad.d2547d81z
# also available at https://zenodo.org/record/4656848
# assumes raw selection tables are saved in raw/selection_tables,
# with the exception of audio files which are saved in soundscape_data.

import pandas as pd
import os
import tqdm
import numpy as np
from glob import glob

def main():
  cwd = os.getcwd()

  raw_data_dir = os.path.join(cwd, 'raw')
  raw_annot_dir = os.path.join(raw_data_dir, 'selection_tables')
  audio_dir = os.path.join(cwd, 'soundscape_data')

  formatted_data_dir = os.path.join(cwd, 'formatted')
  formatted_annot_dir = os.path.join(formatted_data_dir, 'selection_tables')
  for d in [formatted_data_dir, formatted_annot_dir]:
    if not os.path.exists(d):
      os.makedirs(d)

  raw_annotations_fps = sorted(glob(os.path.join(cwd, 'raw', 'selection_tables', '*.txt')))

  train_proportion = 0.6
  val_proportion = 0.2

  train_audio_fps = []
  val_audio_fps = []
  test_audio_fps = []

  for i in range(1,5):
    audio_fps = sorted(glob(os.path.join(audio_dir, f"Recording_{i}_*")))
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
    annot_fn = f"{fn}.Table.1.selections.txt"
    raw_annotations_fp = os.path.join(raw_annot_dir, annot_fn)

    annot_df = pd.read_csv(raw_annotations_fp, sep = '\t')
    annot_df['Annotation'] = annot_df['Species']
    annot_df = annot_df.drop('Species', axis=1)

    annot_fn = f"{fn}.Table.1.selections.txt"
    annot_fp = os.path.join(formatted_annot_dir, annot_fn)

    annot_df.to_csv(annot_fp, sep = '\t', index = False)
    train_annot_fps.append(annot_fp)

  for fn, audio_fp in zip(val_fns, val_audio_fps):
    annot_fn = f"{fn}.Table.1.selections.txt"
    raw_annotations_fp = os.path.join(raw_annot_dir, annot_fn)

    annot_df = pd.read_csv(raw_annotations_fp, sep = '\t')
    annot_df['Annotation'] = annot_df['Species']
    annot_df = annot_df.drop('Species', axis=1)

    annot_fn = f"{fn}.Table.1.selections.txt"
    annot_fp = os.path.join(formatted_annot_dir, annot_fn)

    annot_df.to_csv(annot_fp, sep = '\t', index = False)
    val_annot_fps.append(annot_fp)

  for fn, audio_fp in zip(test_fns, test_audio_fps):
    annot_fn = f"{fn}.Table.1.selections.txt"
    raw_annotations_fp = os.path.join(raw_annot_dir, annot_fn)

    annot_df = pd.read_csv(raw_annotations_fp, sep = '\t')
    annot_df['Annotation'] = annot_df['Species']
    annot_df = annot_df.drop('Species', axis=1)

    annot_fn = f"{fn}.Table.1.selections.txt"
    annot_fp = os.path.join(formatted_annot_dir, annot_fn)

    annot_df.to_csv(annot_fp, sep = '\t', index = False)
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
  breakpoint()

if __name__ == "__main__":
  main()
