from plumbum import local, FG
import os
import yaml
import numpy as np


def main():
  # Project setup
  local['python']['../main.py',
                  'project-setup',
                  '--train-info-fp=/home/jupyter/sound_event_detection/datasets/hawaii/formatted/train_info.csv',
                  '--val-info-fp=/home/jupyter/sound_event_detection/datasets/hawaii/formatted/val_info.csv',
                  '--test-info-fp=/home/jupyter/sound_event_detection/datasets/hawaii/formatted/test_info.csv',
                  '--project-dir=/home/jupyter/sound_event_detection/projects/hawaii_experiment_detectron'] & FG
  
  params_file = '/home/jupyter/sound_event_detection/projects/hawaii_experiment_detectron/project_config.yaml'
  with open(params_file, 'r') as f:
    cfg = yaml.safe_load(f)
    
  new_label_mapping = {}
  
  for k in cfg['label_mapping']:
    if k == 'Unknown':
      new_label_mapping[k] = 'Unknown'
    # omit skylar because very long boxes
    elif k in ['hawama','iiwi','ercfra','apapan','houfin','warwhe1','reblei','omao']:
      new_label_mapping[k] = k
      
  cfg['label_mapping'] = new_label_mapping
  cfg['label_set'] = ['hawama','iiwi','ercfra','apapan','houfin','warwhe1','reblei','omao']
  
  with open(params_file, "w") as f:
    yaml.dump(cfg, f)
  
  j = 0
  for lr in [.005, .001, .0005]:
    for batch_size in [16, 4]:
      try:
        local['python']['../main.py',
                        'train-comparison',
                        '--project-config-fp=/home/jupyter/sound_event_detection/projects/hawaii_experiment_detectron/project_config.yaml',
                        f'--name=m{j}',
                        f'--batch-size={batch_size}',
                        f'--n-epochs=200',
                        '--opts',
                        'SOLVER.BASE_LR',
                        f'{lr}'
                       ] & FG
      except:
        pass
      j += 1

if __name__ == "__main__":
  main()
