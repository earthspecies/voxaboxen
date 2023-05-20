import pandas as pd
import librosa
import os
import soundfile as sf
import tqdm
from pathlib import Path

def modify_anno(anno_fp, modified_anno_fp, start, end):
  anno = pd.read_csv(anno_fp, sep = '\t')
  anno = anno[anno['View'] == 'Waveform 1']
  mask = (anno['Begin Time (s)'] >= start) & (anno['End Time (s)'] < end)
  modified_anno = anno[mask].copy()
  modified_anno['Begin Time (s)'] = modified_anno['Begin Time (s)'] - start
  modified_anno['End Time (s)'] = modified_anno['End Time (s)'] - start
  modified_anno['Annotation'] = "crow"
  modified_anno.to_csv(modified_anno_fp, sep = '\t', index = False)


if __name__ == "__main__":
    ############
    # Settings #
    ############

    out_info_dir = '/home/jupyter/sound_event_detection/datasets/alala/formatted/'
    if not os.path.exists(out_info_dir):
      os.makedirs(out_info_dir)

    # the root of the spanish_carrion_crows dataset
    base_data_dir = '/home/jupyter/sound_event_detection/datasets/alala/raw/'
    base_anno_dir = os.path.join(base_data_dir, 'annotations')
    # anno_dir = os.path.join(base_anno_dir, 'selection_tables')
    
    # Where to save chunked clips and selection tables
    chunked_clips_dir = os.path.join(out_info_dir, 'chunked_clips')
    if not os.path.exists(chunked_clips_dir):
      os.makedirs(chunked_clips_dir)
    modified_anno_dir = os.path.join(out_info_dir, 'modified_selection_tables')
    if not os.path.exists(modified_anno_dir):
      os.makedirs(modified_anno_dir)
    
    train_chunk_dur_sec = 60*5
    val_chunk_dur_sec = 60*0

    ########
    # Train and Val
    ########
        
    split = 'dev'
    
    dev_split =['KBCC_Aviary_4', 'KBCC_Hala', 'MBCC_Alani', 'MBCC_Lowers1']
    
    # split_fp = os.path.join(split_dir, f'{split}.txt')
    
    train_out_info = pd.DataFrame(columns=['fn', 'audio_fp', 'selection_table_fp'])
    val_out_info = pd.DataFrame(columns=['fn', 'audio_fp', 'selection_table_fp'])
    
    train_out_info_fp = os.path.join(out_info_dir, f'train_info_iter_0.csv')
    val_out_info_fp = os.path.join(out_info_dir, f'val_info.csv')
    
    for aviary in dev_split:
      filepaths = Path(base_data_dir, 'crows_subselected', aviary).glob('*.wav')
      for audio_fp in tqdm.tqdm(filepaths):
          duration = librosa.get_duration(filename=audio_fp)
          audio, sr = librosa.load(audio_fp, sr=None)
          
          fn = f"{aviary}.{audio_fp.stem}"
          
          anno_fp = os.path.join(base_anno_dir, aviary, f'{audio_fp.stem}.Table.1.selections.txt')

          t = 0
          i = 0
          while t<duration:
            end = min(t+train_chunk_dur_sec, duration)

            modified_anno_fp = os.path.join(modified_anno_dir, f'{fn}.{i}.txt')
            modify_anno(anno_fp, modified_anno_fp, t, end)

            start_sample = int(t * sr)
            end_sample = int(end * sr)
            chunked_audio_fp = os.path.join(chunked_clips_dir, f"{fn}.{i}.wav")

            sf.write(chunked_audio_fp, audio[start_sample:end_sample], sr)

            train_out_info = train_out_info.append({'fn': f'{fn}.{i}', 'audio_fp': chunked_audio_fp, 'selection_table_fp': modified_anno_fp}, ignore_index=True)

            t += train_chunk_dur_sec
            i += 1

            if t >= duration:
              break

            end = min(t+val_chunk_dur_sec, duration)
            modified_anno_fp = os.path.join(modified_anno_dir, f'{fn}.{i}.txt')
            modify_anno(anno_fp, modified_anno_fp, t, end)

            start_sample = int(t * sr)
            end_sample = int(end * sr)
            chunked_audio_fp = os.path.join(chunked_clips_dir, f"{fn}.{i}.wav")

            sf.write(chunked_audio_fp, audio[start_sample:end_sample], sr)

            val_out_info = val_out_info.append({'fn': f'{fn}.{i}', 'audio_fp': chunked_audio_fp, 'selection_table_fp': modified_anno_fp}, ignore_index=True)

            t += val_chunk_dur_sec
            i += 1

    train_out_info.to_csv(train_out_info_fp, index=False)
    val_out_info.to_csv(val_out_info_fp, index=False)

    ########
    # Test
    ########
    
    # We do not chunk up test clips
    
    split = 'test'
    
    test_split =['KBCC_Ohelo', 'MBCC_Makani2']
    
    test_out_info_fp = os.path.join(out_info_dir, f'test_info.csv')
    test_out_info = pd.DataFrame(columns=['fn', 'audio_fp', 'selection_table_fp'])
    
    # split_fp = os.path.join(split_dir, f'{split}.txt')
    # fns = pd.read_csv(split_fp, header=None)[0].to_list()
    
    for aviary in test_split:
      filepaths = Path(base_data_dir, 'crows_subselected', aviary).glob('*.wav')
      for audio_fp in tqdm.tqdm(filepaths):
          fn = f"{aviary}.{audio_fp.stem}"

          chunked_audio_fp = os.path.join(chunked_clips_dir, f"{fn}.wav")
          audio, sr = librosa.load(audio_fp, sr=None)

          anno_fp = os.path.join(base_anno_dir, aviary, f'{audio_fp.stem}.Table.1.selections.txt')

          modified_anno_fp = os.path.join(modified_anno_dir, f'{fn}.txt')
          duration = librosa.get_duration(filename=audio_fp)
          modify_anno(anno_fp, modified_anno_fp, 0, duration)
          sf.write(chunked_audio_fp, audio, sr)

          test_out_info = test_out_info.append({'fn': fn, 'audio_fp': chunked_audio_fp, 'selection_table_fp': modified_anno_fp}, ignore_index=True)

    test_out_info.to_csv(test_out_info_fp, index=False)

    ########
    # Unannotated
    ########
    
    # We create a manifest of unannotated files which will be sampled during active learning
    # We do not resample them now.
    
    split = 'unannotated'
    
    # split_fp = os.path.join(split_dir, f'{split}.txt')
    
    unannotated_split = ['KBCC_Aviary_1']
    unannotated_out_info_fp = os.path.join(out_info_dir, f'train_pool_info.csv')

    unannotated_out_info = pd.DataFrame(columns=['fn', 'audio_fp', 'selection_table_fp'])
    
    for aviary in unannotated_split:
      audio_fps = sorted(Path(base_data_dir, aviary).glob('*.wav'))
      fns = [f"{aviary}.{x.stem}" for x in audio_fps]
      
      for audio_fp, fn in zip(audio_fps, fns):
        unannotated_out_info = unannotated_out_info.append({'fn': fn, 'audio_fp': str(audio_fp), 'selection_table_fp' : "None"}, ignore_index=True)
    unannotated_out_info.to_csv(unannotated_out_info_fp, index=False)
