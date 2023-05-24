import pandas as pd
import librosa
import os
import soundfile as sf
import tqdm
from intervaltree import IntervalTree

def modify_anno(anno_fp, modified_anno_fp, start, end):
  anno = pd.read_csv(anno_fp, sep = '\t')
  mask = (anno['Begin Time (s)'] >= start) & (anno['End Time (s)'] < end)
  modified_anno = anno[mask].copy()
  modified_anno['Begin Time (s)'] = modified_anno['Begin Time (s)'] - start
  modified_anno['End Time (s)'] = modified_anno['End Time (s)'] - start
  modified_anno = modified_anno.reset_index()
  
  merged_anno = merge_annos(modified_anno, merge_tolerance = 0.5)
  merged_anno.to_csv(modified_anno_fp, sep = '\t', index = False)
  
def merge_annos(df, merge_tolerance = 0):
  classes = list(df['Annotation'].unique())
  
  df['End Time (s)'] += merge_tolerance
  
  begin_times = []
  end_times = []
  annotations = []
  
  for c in classes:
    df_sub = df[df['Annotation'] == c]
    
    # form interval tree of annos
    tree = IntervalTree()
    for i, row in df_sub.iterrows():
      begin = row['Begin Time (s)']
      end = row['End Time (s)']
      if end>begin:
        tree[begin:end] = c
     
    # merge overlaps
    tree.merge_overlaps(strict=False)
    
    #add to output dataframe
    for x in tree:
      begin_times.append(x.begin)
      end_times.append(x.end)
      annotations.append(c)
    
  output_df = pd.DataFrame({'Begin Time (s)' : begin_times, 'End Time (s)' : end_times, 'Annotation' : annotations})
  
  output_df['End Time (s)'] -= merge_tolerance
  return output_df

if __name__ == "__main__":
    ############
    # Settings #
    ############

    out_info_dir = '/home/jupyter/carrion_crows_data/call_detection_data.merge_boxes/'

    # the root of the spanish_carrion_crows dataset
    base_data_dir = '/home/jupyter/carrion_crows_data/'
    base_anno_dir = os.path.join(base_data_dir, 'Annotations_revised_by_Daniela.cleaned')
    split_dir = os.path.join(base_anno_dir, 'split')
    anno_dir = os.path.join(base_anno_dir, 'selection_tables')
    
    # Where to save chunked clips and selection tables
    chunked_clips_dir = os.path.join(out_info_dir, 'chunked_clips')
    if not os.path.exists(chunked_clips_dir):
      os.makedirs(chunked_clips_dir)
    modified_anno_dir = os.path.join(out_info_dir, 'modified_selection_tables')
    if not os.path.exists(modified_anno_dir):
      os.makedirs(modified_anno_dir)
    
    train_chunk_dur_sec = 60*54
    val_chunk_dur_sec = 60*6

    ########
    # Train and Val
    ########
        
    split = 'dev'
    
    split_fp = os.path.join(split_dir, f'{split}.txt')
    fns = pd.read_csv(split_fp, header=None)[0].to_list()
    train_out_info_fp = os.path.join(out_info_dir, f'train_info_iter_0.csv')
    val_out_info_fp = os.path.join(out_info_dir, f'val_info.csv')

    train_out_info = pd.DataFrame(columns=['fn', 'audio_fp', 'selection_table_fp'])
    val_out_info = pd.DataFrame(columns=['fn', 'audio_fp', 'selection_table_fp'])

    
    for fn in tqdm.tqdm(fns):
        audio_fp = os.path.join(base_data_dir, '/'.join(fn.split('_')) + '.wav')
        duration = librosa.get_duration(filename=audio_fp)
        audio, sr = librosa.load(audio_fp, sr=None)
        
        anno_fp = os.path.join(anno_dir, f'{fn}.txt')

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
    
    split_fp = os.path.join(split_dir, f'{split}.txt')
    fns = pd.read_csv(split_fp, header=None)[0].to_list()
    test_out_info_fp = os.path.join(out_info_dir, f'test_info.csv')

    test_out_info = pd.DataFrame(columns=['fn', 'audio_fp', 'selection_table_fp'])
    
    for fn in tqdm.tqdm(fns):
        audio_fp = os.path.join(base_data_dir, '/'.join(fn.split('_')) + '.wav')
        chunked_audio_fp = os.path.join(chunked_clips_dir, f"{fn}.wav")
        audio, sr = librosa.load(audio_fp, sr=None)
        
        anno_fp = os.path.join(anno_dir, f'{fn}.txt')

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
    
    split_fp = os.path.join(split_dir, f'{split}.txt')
    fns = pd.read_csv(split_fp, header=None)[0].to_list()
    unannotated_out_info_fp = os.path.join(out_info_dir, f'train_pool_info.csv')

    unannotated_out_info = pd.DataFrame(columns=['fn', 'audio_fp', 'selection_table_fp'])
    
    for fn in fns:
      audio_fp = os.path.join(base_data_dir, '/'.join(fn.split('_')) + '.wav')

      unannotated_out_info = unannotated_out_info.append({'fn': fn, 'audio_fp': audio_fp, 'selection_table_fp' : "None"}, ignore_index=True)
    unannotated_out_info.to_csv(unannotated_out_info_fp, index=False)
