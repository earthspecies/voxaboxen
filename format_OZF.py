import os
from natsort import natsorted
import json
import shutil
import pandas as pd


selection_table_fns = natsorted([x for x in os.listdir('ZFdset') if x.endswith('.txt')])

raw_to_formatted_map = {}
split_infos = {'train':[], 'val':[], 'test':[]}
dset_dir = 'OZF/formatted'
os.makedirs(f'{dset_dir}/selection_tables', exist_ok=True)
os.makedirs(f'{dset_dir}/audio', exist_ok=True)
for i,raw_st_fn in enumerate(selection_table_fns):
    if i < len(selection_table_fns)*7/10:
        split='train'
    elif i < len(selection_table_fns)*8/10:
        split='val'
    else:
        split='test'
    number = int(raw_st_fn.split('.')[0])
    new_fn = f'OZF{i}_{split}'
    raw_audio_fn = f'{number}.flac'
    raw_to_formatted_map[i] = (raw_st_fn, new_fn)
    assert os.path.exists(raw_st_fp:=f'ZFdset/{raw_st_fn}')
    assert os.path.exists(raw_audio_fp:=f'ZFdset/audio/{raw_audio_fn}')
    shutil.copy(raw_st_fp, new_st_fp:=f'{dset_dir}/selection_tables/{new_fn}.txt')
    shutil.copy(raw_audio_fp, new_audio_fp:=f'{dset_dir}/audio/{new_fn}.flac')
    print(split)
    split_infos[split].append({'fn':new_fn, 'audio_fp':new_audio_fp, 'selection_table_fp':new_st_fp})

for k,v in split_infos.items():
    df = pd.DataFrame(v)
    df.to_csv(f'{dset_dir}/{k}_info.csv')

breakpoint()
with open(f'{dset_dir}/raw_to_formatted_map.json', 'w') as f:
    json.dump(raw_to_formatted_map, f)
