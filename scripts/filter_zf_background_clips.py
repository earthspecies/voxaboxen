import pandas as pd
import os
from glob import glob
from tqdm import tqdm

audio_fps = sorted(glob('/home/jupyter/data/voxaboxen_data/zebra_finch_synthetic/raw/background/Stereo/*.wav'))
audio_fps_filtered = []

light_on = int(6.5*60*60*1000)
light_off = int(20.5*60*60*1000)
bad_files = 0

for audio_fp in tqdm(audio_fps):
    fn = os.path.basename(audio_fp).split('.')[0]
    time = int(fn.split('_')[-1])
    if (time<light_on) or (time>light_off):
        continue
    st_fp = os.path.join('/home/jupyter/data/voxaboxen_data/zebra_finch_synthetic/raw/background/Spectral_features_tables', "peaks_pred_" + fn + ".txt")
    if not os.path.exists(st_fp):
        print(f"Could not find {st_fp}")
        continue
    st = pd.read_csv(st_fp, sep='\t')
    if len(st) == 0:
        audio_fps_filtered.append(audio_fp)
    else:
        bad_files += 1
        
audio_fps_filtered = pd.DataFrame({"audio_fp" : audio_fps_filtered})
print(f"Found {len(audio_fps)} acceptable files, and there were {bad_files} files with vox in them")
audio_fps_filtered.to_csv('/home/jupyter/data/voxaboxen_data/zebra_finch_synthetic/raw/background/audio_fps_filtered.csv')
