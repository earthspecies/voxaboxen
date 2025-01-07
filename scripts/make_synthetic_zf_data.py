import soundfile as sf
import librosa
import numpy as np
import os
import pandas as pd
from glob import glob
from tqdm import tqdm

vox_fps = pd.read_csv('/home/jupyter/data/voxaboxen_data/zebra_finch_synthetic/filtered_zf_vox.csv')
vox_fps_sorted = sorted(vox_fps[vox_fps['include']]['audio_fp'])
print(len(vox_fps_sorted))
background_fps_sorted = sorted(pd.read_csv('/home/jupyter/data/voxaboxen_data/zebra_finch_synthetic/raw/background/audio_fps_filtered.csv')['audio_fp'])
output_parent_dir = '/home/jupyter/data/voxaboxen_data/OZF_synthetic'

target_overlap_percs = [0, 0.2, 0.4, 0.6, 0.8, 1]
dur_clip = 60

orig_sts = sorted(glob('/home/jupyter/data/voxaboxen_data/OZF/formatted/selection_tables/*.txt'))

clipnames_to_n_events = {}

for orig_st in orig_sts:
    st = pd.read_csv(orig_st,sep='\t')
    n_vox = len(set(st['Selection']))
    clipname = os.path.basename(orig_st).replace(".txt", "")
    clipnames_to_n_events[clipname] = n_vox
    print(f"{clipname} has {n_vox} vox")
    
clipnames = sorted(clipnames_to_n_events.keys())

def compute_overlap_perc(st):
    st = pd.DataFrame(st)
    n_vox = len(st)
    if n_vox == 0:
        return 0
    # n_vox_with_overlap = 0
    n_overlaps = 0
    for i, row in st.iterrows():
        b = row["Begin Time (s)"]
        e = row["End Time (s)"]
        n_overlaps += len(st[(st["Begin Time (s)"] >= b) & (st["Begin Time (s)"] < e) & (st["End Time (s)"] > e)])
        n_overlaps += len(st[(st["Begin Time (s)"] > b) & (st["End Time (s)"] <= e)])
        
        
        # if len(st[(st["Begin Time (s)"] < b) & (st["End Time (s)"] > b)])>0:
        #     n_vox_with_overlap += 1
        #     continue
        # if len(st[(st["Begin Time (s)"] < e) & (st["End Time (s)"] > e)])>0:
        #     n_vox_with_overlap += 1
        #     continue
        # if len(st[(st["Begin Time (s)"] > b) & (st["End Time (s)"] < e)])>0:
        #     n_vox_with_overlap += 1
        #     continue
    # return n_vox_with_overlap/n_vox
    return n_overlaps/n_vox

def generate_scene(events, background_audio, overlap_perc, sr, rng, init_percent=0.25, pad_dur=0.05):
    pad_samples = int(sr*pad_dur)
    
    background_rms = np.std(background_audio)
    
    current_overlap_perc = 1
    vox_mask = np.zeros_like(background_audio)
    st = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}
    if overlap_perc > init_percent:
        n_init_events = int(len(events) * init_percent)
    else:
        n_init_events = 0
    
    for event_number, event in tqdm(enumerate(events)):
        event_dur_samples = max(0, len(event)-2*pad_samples)
        if event_number<n_init_events:
            idx_to_place_event = rng.integers(0,len(background_audio))
         
        else:
            event_mask_shifted = vox_mask
            for shift in range(1,event_dur_samples,200):
                shifted_mask = np.concatenate([vox_mask[shift:], np.zeros((shift,))])
                event_mask_shifted = np.maximum(event_mask_shifted,shifted_mask)
            
            if current_overlap_perc >= overlap_perc:
                # find a place to put the vox where it is not overlapping
                idxs_where_results_in_no_overlap = np.nonzero(1-event_mask_shifted)[0]
                if len(idxs_where_results_in_no_overlap)>0:
                    idx_to_place_event = rng.permutation(idxs_where_results_in_no_overlap)[0]
                else:
                    idx_to_place_event = rng.integers(0,len(background_audio))
            else:
                # find a place to put the vox where it is overlapping
                idxs_where_results_in_overlap = np.nonzero(event_mask_shifted)[0]
                if len(idxs_where_results_in_overlap)>0:
                    idx_to_place_event = rng.permutation(idxs_where_results_in_overlap)[0]
                else:
                    idx_to_place_event = rng.integers(0,len(background_audio))
                
        # place event
        try:
            event_rms = np.std(event[pad_samples:-pad_samples])
        except:
            print("padding issue")
            event_rms = 0
        desired_snr_dB = rng.uniform(low=-15,high=0)
        event = event * (background_rms / event_rms) * (10**(.05 * desired_snr_dB))
        
        idx_to_place_event_corrected = max(0, idx_to_place_event - pad_samples)
        background_audio[idx_to_place_event_corrected:idx_to_place_event_corrected+len(event)] += event[:len(background_audio[idx_to_place_event_corrected:idx_to_place_event_corrected+len(event)])] # adjust for pad
        
        vox_mask[idx_to_place_event:idx_to_place_event+event_dur_samples] = np.maximum(vox_mask[idx_to_place_event:idx_to_place_event+event_dur_samples], np.ones_like(vox_mask[idx_to_place_event:idx_to_place_event+event_dur_samples]))
        
        st["Begin Time (s)"].append(idx_to_place_event/sr)
        st["End Time (s)"].append(min(len(background_audio), idx_to_place_event+event_dur_samples)/sr)
        st["Annotation"].append("POS")
        
        current_overlap_perc = compute_overlap_perc(st)
        
    print(current_overlap_perc)
    st = pd.DataFrame(st)
    return background_audio, st

for overlap_perc in target_overlap_percs:
    # train_info = {}
    # val_info = {}
    # test_info = {}
    data_info = {"fn" : [], "audio_fp" : [], "selection_table_fp" : []}
    
    output_audio_dir = os.path.join(output_parent_dir, f"overlap_{overlap_perc}", "audio")
    output_st_dir = os.path.join(output_parent_dir, f"overlap_{overlap_perc}", "selection_tables")
    for dname in [output_audio_dir, output_st_dir]:
        if not os.path.exists(dname):
            os.makedirs(dname)
    
    rng = np.random.default_rng(0)
    vox_fps = list(rng.permutation(vox_fps_sorted))
    background_fps = list(rng.permutation(background_fps_sorted))
    
    for clipname in clipnames:
        n_events = clipnames_to_n_events[clipname]
        
        background_fp = background_fps.pop()
        background_audio, sr = librosa.load(background_fp, mono=True, sr=16000)
        while len(background_audio)/sr < dur_clip:
            next_background_fp = background_fps.pop()
            next_background_audio, sr = librosa.load(next_background_fp, mono=True, sr=16000)
            background_audio = np.concatenate([background_audio, next_background_audio])
            
        background_audio = background_audio[:int(dur_clip*sr)]
        
        events = []
        for _ in range(n_events):
            event_fp = vox_fps.pop()
            event, sr = librosa.load(event_fp, mono=True, sr=16000)
            events.append(event)
            
        scene, st = generate_scene(events, background_audio, overlap_perc, sr, rng)
        
        audio_target_fp = os.path.join(output_audio_dir, f"{clipname}.wav")
        sf.write(audio_target_fp, scene, sr)
        
        st_target_fp = os.path.join(output_st_dir, f"{clipname}.txt")
        st.to_csv(st_target_fp, sep='\t', index=False)
        
        data_info["fn"].append(clipname)
        data_info["audio_fp"].append(audio_target_fp)
        data_info["selection_table_fp"].append(st_target_fp)
        
    data_info = pd.DataFrame(data_info)
    info_target_fp = os.path.join(output_parent_dir, f"overlap_{overlap_perc}", "data_info.csv")
    data_info.to_csv(info_target_fp, index=False)
