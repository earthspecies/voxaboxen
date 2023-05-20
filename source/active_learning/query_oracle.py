import pandas as pd
import os

def query_oracle(al_args, output_log):
  train_pool_info_fp = al_args.train_pool_info_fp
  train_pool_info = pd.read_csv(train_pool_info_fp)
  
  begin_times = []
  end_times = []
  annotations = []
  
  for i, row in output_log.iterrows():
    start_second = row['start_second'] # start second in original file
    clip_duration = row['duration']
    end_second = clip_duration + start_second # end second in original file
    filename = row['fn']
    start_second_in_al_samples = row['start_second_in_al_samples'] # start second in clip output by active learning
    
    anno_row = train_pool_info[train_pool_info['fn'] == filename]
    anno_fp = str(anno_row['selection_table_fp'].unique()[0])
    assert anno_fp != "None", "annotation does not exist"
    
    anno = pd.read_csv(anno_fp, delimiter = '\t')
    anno_sub = anno[(anno['End Time (s)'] >= start_second) & (anno['Begin Time (s)']<=end_second)]
    for j, selection in anno_sub.iterrows():
      # Set annotations for partial boxes to be unknown
      begin_relative_to_sampled_clip = selection['Begin Time (s)'] - start_second
      end_relative_to_sampled_clip = selection['End Time (s)'] - start_second
      annotation = selection['Annotation']
      if begin_relative_to_sampled_clip < 0:
        begin_relative_to_sampled_clip = 0
        annotation = "Unknown"
      if end_relative_to_sampled_clip > clip_duration:
        end_relative_to_sampled_clip = clip_duration
        annotation = "Unknown"
        
      # Adjust to time in clip output by active learning
      begin_relative_to_output = begin_relative_to_sampled_clip + start_second_in_al_samples
      end_relative_to_output = end_relative_to_sampled_clip + start_second_in_al_samples
      
      begin_times.append(begin_relative_to_output)
      end_times.append(end_relative_to_output)
      annotations.append(annotation)
      
  selection_table = pd.DataFrame({'Begin Time (s)' : begin_times, 'End Time (s)' : end_times, 'Annotation' : annotations}) 
  
  selection_table_fp = os.path.join(al_args.output_dir, f"selection_table_{al_args.name}.txt")
  selection_table.to_csv(selection_table_fp, sep = '\t', index = False)
  return selection_table_fp
  