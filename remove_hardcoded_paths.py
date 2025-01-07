import os
import pandas as pd
import re


def fix(s):
    s = s.removeprefix('/home/jupyter/data/voxaboxen_data/')
    if not s.startswith('datasets/'):
        s = 'datasets/' + s
    if 'slowed' in s:
        print(s, re.sub(r'slowed_[0-9\.]*', 'slowed', s))
    s = re.sub(r'slowed_[0-9\.]*', 'slowed', s)
    return s

def df_maybe_remove(fp):
    df = pd.read_csv(fp, index_col=0)
    df['selection_table_fp'] = [fix(x) for x in df['selection_table_fp']]
    df['audio_fp'] = [fix(x) for x in df['audio_fp']]
    df.to_csv(fp)

for dset_name in os.listdir('datasets/'):
    for split in ('train', 'val', 'test'):
        if dset_name=='OZF_synthetic':
            continue
        subdirs = os.listdir('datasets/OZF_synthetic') if dset_name == 'OZF_synthetic' else ['formatted']
        for sd in subdirs:
            fp = f'datasets/{dset_name}/{sd}/{split}_info.csv'
            df_maybe_remove(fp)
    slowed_ratio_remove_name = re.sub(r'slowed_[0-9\.]*', 'slowed', dset_name)
    os.rename(f'datasets/{dset_name}', f'datasets/{slowed_ratio_remove_name}')
