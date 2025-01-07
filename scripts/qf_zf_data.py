from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from glob import glob
import soundfile as sf
import numpy as np
import os
import shutil
import pandas as pd
from tqdm import tqdm

n_required = 18000
vox_fps_sorted = sorted(glob('/home/jupyter/data/voxaboxen_data/zebra_finch_synthetic/raw/events/**/*.wav'))
output = {"audio_fp" : [], "include" : []}
analyzer = Analyzer()

n_accepted = 0

for fp in tqdm(vox_fps_sorted):
    output["audio_fp"].append(fp)
    
    audio, sr = sf.read(fp)
    
    pad_len = int(.75 * sr)
    audio_padded = np.pad(audio, (pad_len, pad_len))
    
    sf.write('temp.wav', audio_padded, sr)
    
    recording = Recording(
        analyzer,
        'temp.wav',
        # lat=35.4244,
        # lon=-120.7463,
        # date=datetime(year=2022, month=5, day=10), # use date or week_48
        min_conf=0,
    )
    recording.analyze()
    
    if len(recording.detections) == 0:
        output["include"].append(False)
        
    else:
        iszf = False
        for x in recording.detections:
            if "Finch" in x['common_name']:
                iszf = True
                n_accepted += 1
        output["include"].append(iszf)
    
    if n_accepted >= n_required:
       break
    
output = pd.DataFrame(output)
output.to_csv("/home/jupyter/data/voxaboxen_data/zebra_finch_synthetic/filtered_zf_vox.csv")
print(output["include"].value_counts())