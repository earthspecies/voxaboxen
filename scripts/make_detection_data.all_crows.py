import pandas as pd
import librosa
import os


def process_anno(fn, anno_dir, out_ts_dir, label_mapping):
    anno_fp = os.path.join(anno_dir, f'{fn}.txt')
    anno = pd.read_csv(anno_fp, sep='\t')

    ts_df = pd.DataFrame(columns=['start', 'end', 'label'])
    for i, row in anno.iterrows():
        start = row['Begin Time (s)']
        end = row['End Time (s)']
        label = row['Annotation']

        if end <= start:
            continue

        if label in label_mapping:
            ts_df = ts_df.append({'start': start, 'end': end, 'label': label_mapping[label]}, ignore_index=True)

    out_ts_fp = os.path.join(out_ts_dir, f'{fn}.csv')
    ts_df.to_csv(out_ts_fp, index=False)

    return out_ts_fp


if __name__ == "__main__":
    ############
    # Settings #
    ############

    label_mapping = {
        'focal': 'crow',
        'focal?': 'crow',
        'not focal': 'crow',
        'not focal LD': 'crow',
        'not focal?': 'crow',
        'crowchicks': 'crow',
        'crow_undeter': 'crow',
        'nest': 'crow',
    }

    # out_info_dir = '/home/jupyter/storage/Datasets/spanish_carrion_crows/call_detection_data.revised_anno.all_crows/'
    out_info_dir = '/home/jupyter/carrion_crows_data/call_detection_data.revised_anno.all_crows/'

    # the root of the spanish_carrion_crows dataset
    # base_data_dir = '/home/jupyter/storage/Datasets/spanish_carrion_crows/raw/'
    base_data_dir = '/home/jupyter/carrion_crows_data/'

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
        for fn in fns:
            audio_fp = os.path.join(base_data_dir, '/'.join(fn.split('_')) + '.wav')
            duration = librosa.get_duration(filename=audio_fp)

            ts_fp = process_anno(fn, anno_dir, out_ts_dir, label_mapping)

            out_info = out_info.append({'fn': fn, 'duration': duration, 'audio_fp': audio_fp, 'timestamp_fp': ts_fp}, ignore_index=True)

        out_info.to_csv(out_info_fp, index=False)
