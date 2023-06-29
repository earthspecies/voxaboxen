from plumbum import local, FG
import os
import yaml
import numpy as np


def main():
  # Project setup
  local['python']['../main.py',
                  'project-setup',
                  '--train-pool-info-fp=/home/jupyter/sound_event_detection/datasets/powdermill/formatted/train_pool_info.csv',
                  '--test-info-fp=/home/jupyter/sound_event_detection/datasets/powdermill/formatted/test_info.csv',
                  '--project-dir=/home/jupyter/sound_event_detection/projects/powdermill_experiment_coreset'] & FG
  
  params_file = '/home/jupyter/sound_event_detection/projects/powdermill_experiment_coreset/project_config.yaml'
  with open(params_file, 'r') as f:
    cfg = yaml.safe_load(f)
    
  new_label_mapping = {}
  
  for k in cfg['label_mapping']:
    if k == 'Unknown':
      new_label_mapping[k] = 'Unknown'
    else:
      new_label_mapping[k] = 'voc'
      
  cfg['label_mapping'] = new_label_mapping
  cfg['label_set'] = ['voc']
  
  with open(params_file, "w") as f:
    yaml.dump(cfg, f)
  
  for i in range(20):
    if i == 0:
      local['python']['../main.py',
                      'active-learning-sampling',
                      '--project-config-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment_coreset/project_config.yaml',
                      '--sampling-method=coreset',
                      '--sample-duration=10', 
                      '--sequence-name=core',
                      '--query-oracle',
                      '--max-n-clips-to-sample=50'] & FG
    else:
      local['python']['../main.py',
                      'active-learning-sampling',
                      '--project-config-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment_coreset/project_config.yaml',
                      '--sampling-method=coreset',
                      '--sample-duration=10', 
                      f'--prev-iteration-info-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment_coreset/active_learning/train_info_core_{i-1}.csv',
                      f'--model-args-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment_coreset/m{i-1}/params.yaml',
                      '--query-oracle',
                      '--max-n-clips-to-sample=50'] & FG
      
    local['python']['../main.py',
                    'train-model',
                    '--project-config-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment_coreset/project_config.yaml',
                    f'--name=m{i}',
                    '--clip-duration=10',
                    '--clip-hop=5',
                    '--batch-size=8',
                    '--omit-empty-clip-prob=0',
                    f'--train-info-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment_coreset/active_learning/train_info_core_{i}.csv'
                   ] & FG   

if __name__ == "__main__":
  main()
