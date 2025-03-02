# Script to process data from https://zenodo.org/records/8342596

# To get the data
# cd datasets/AnuraSet
# wget "https://zenodo.org/records/8342596/files/raw_data.zip?download=1" -O raw_data.zip
# unzip raw_data.zip
# rm raw_data.zip
# wget "https://zenodo.org/records/8342596/files/strong_labels.zip?download=1" -O strong_labels.zip
# unzip strong_labels.zip
# rm strong_labels.zip


import pandas as pd
import os
import tqdm
import numpy as np
from glob import glob

def read_selection_table(filepath):
    """Read AnuraSet selection table format and convert to standard format"""
    df = pd.read_csv(filepath, sep='\t', header=None, names=['Begin Time (s)', 'End Time (s)', 'Annotation'],
                     usecols=[0,1,2])
    # Remove quality suffixes (_L, _M, _H) from species codes
    df['Annotation'] = df['Annotation'].str.replace(r'_(L|M|H)$', '', regex=True)
    # Add frequency columns (only needed to work with detectron commparison code)
    df['Low Freq (Hz)'] = 0
    df['High Freq (Hz)'] = 10000
    return df

def main():
    cwd = os.getcwd()

    raw_data_dir = os.path.join(cwd, 'raw_data')
    labels_dir = os.path.join(cwd, 'power_labels')

    formatted_data_dir = os.path.join(cwd, 'formatted')
    formatted_annot_dir = os.path.join(formatted_data_dir, 'selection_tables')
    for d in [formatted_data_dir, formatted_annot_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    audio_fps = []
    annot_fps = []

    # AnuraSet has 4 recording sites: INCT4, INCT17, INCT41, INCT20955
    for site in ['INCT4', 'INCT17', 'INCT41', 'INCT20955']:
        site_audio_fps = sorted(glob(os.path.join(raw_data_dir, site, '*.wav')))
        for audio_fp in site_audio_fps:
            fn = os.path.splitext(os.path.basename(audio_fp))[0]
            annot_fp = os.path.join(labels_dir, site, f"{fn}.txt")
            if os.path.exists(annot_fp):
                audio_fps.append(audio_fp)
                annot_fps.append(annot_fp)

    train_proportion = 0.6
    val_proportion = 0.2

    n_files = len(audio_fps)
    n_train = int(train_proportion * n_files)
    n_val = int(val_proportion * n_files)

    np.random.seed(42)
    indices = np.random.permutation(n_files)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]

    train_audio_fps = [audio_fps[i] for i in train_indices]
    train_annot_fps = [annot_fps[i] for i in train_indices]

    val_audio_fps = [audio_fps[i] for i in val_indices]
    val_annot_fps = [annot_fps[i] for i in val_indices]

    test_audio_fps = [audio_fps[i] for i in test_indices]
    test_annot_fps = [annot_fps[i] for i in test_indices]

    for split_name, audio_fps, annot_fps in [
        ('train', train_audio_fps, train_annot_fps),
        ('val', val_audio_fps, val_annot_fps),
        ('test', test_audio_fps, test_annot_fps)
    ]:
        info_rows = []

        for audio_fp, annot_fp in zip(audio_fps, annot_fps):
            fn = os.path.splitext(os.path.basename(audio_fp))[0]
            annot_df = read_selection_table(annot_fp)
            formatted_annot_fn = f"selection_table_{fn}.txt"
            formatted_annot_fp = os.path.join(formatted_annot_dir, formatted_annot_fn)
            annot_df.to_csv(formatted_annot_fp, sep='\t', index=False)

            info_rows.append({
                'fn': fn,
                'audio_fp': audio_fp,
                'selection_table_fp': formatted_annot_fp
            })

        info_df = pd.DataFrame(info_rows)
        info_fp = os.path.join(formatted_data_dir, f'{split_name}_info.csv')
        info_df.to_csv(info_fp, index=False)

if __name__ == "__main__":
    main()