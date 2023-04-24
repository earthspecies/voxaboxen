import pandas as pd
import librosa
import os
import soundfile as sf
import tqdm

def process_anno(fn, anno_dir, out_ts_dir):
    anno_fp = os.path.join(anno_dir, f'{fn}.txt')
    anno = pd.read_csv(anno_fp, sep='\t')

    ts_df = pd.DataFrame(columns=['start', 'end', 'label'])
    for i, row in anno.iterrows():
        start = row['Begin Time (s)']
        end = row['End Time (s)']
        label = row['Annotation']

        if end <= start:
            continue

        ts_df = ts_df.append({'start': start, 'end': end, 'label': label}, ignore_index=True)

    out_ts_fp = os.path.join(out_ts_dir, f'{fn}.csv')
    ts_df.to_csv(out_ts_fp, index=False)

    return out_ts_fp


if __name__ == "__main__":
    ############
    # Settings #
    ############

    # out_info_dir = '/home/jupyter/storage/Datasets/spanish_carrion_crows/call_detection_data.revised_anno.all_crows/'
    out_info_dir = '/home/jupyter/carrion_crows_data/call_detection_data.revised_anno.all/'

    # the root of the spanish_carrion_crows dataset
    # base_data_dir = '/home/jupyter/storage/Datasets/spanish_carrion_crows/raw/'
    base_data_dir = '/home/jupyter/carrion_crows_data/'
    resampled_clips_dir = os.path.join(base_data_dir, 'resampled_clips')
    if not os.path.exists(resampled_clips_dir):
      os.makedirs(resampled_clips_dir)

    # ### Main ###
    # base_anno_dir = '/home/jupyter/storage/Datasets/spanish_carrion_crows/raw/Annotations_revised_by_Daniela.cleaned/'
    base_anno_dir = os.path.join(base_data_dir, 'Annotations_revised_by_Daniela.cleaned')
    split_dir = os.path.join(base_anno_dir, 'split')
    anno_dir = os.path.join(base_anno_dir, 'selection_tables')

    out_ts_dir = os.path.join(out_info_dir, 'timestamp')
    os.makedirs(out_ts_dir, exist_ok=True)

    ########
    # Process
    ########

    # ### output ###
    for split in ['dev', 'test']:
        split_fp = os.path.join(split_dir, f'{split}.txt')
        fns = pd.read_csv(split_fp, header=None)[0].to_list()
        out_info_fp = os.path.join(out_info_dir, f'{split}_info.csv')

        out_info = pd.DataFrame(columns=['fn', 'duration', 'audio_fp', 'timestamp_fp'])
        for fn in tqdm.tqdm(fns):
            ## Omitting because audio was already resampled
            audio_fp = os.path.join(base_data_dir, '/'.join(fn.split('_')) + '.wav')
            duration = librosa.get_duration(filename=audio_fp)
            sr= 16000
            # audio, sr = librosa.load(audio_fp, sr=16000)
            resampled_audio_fp = os.path.join(resampled_clips_dir, f"{fn}.wav")
            # sf.write(resampled_audio_fp, audio, sr)

            ts_fp = process_anno(fn, anno_dir, out_ts_dir)

            out_info = out_info.append({'fn': fn, 'duration': duration, 'audio_fp': resampled_audio_fp, 'timestamp_fp': ts_fp}, ignore_index=True)

        out_info.to_csv(out_info_fp, index=False)
