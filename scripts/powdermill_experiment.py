from plumbum import local, FG
import os
import yaml
import numpy as np


def main():
  # Project setup
  local['python']['../main.py',
                  'project-setup',
                  '--train-info-fp=/home/jupyter/sound_event_detection/datasets/powdermill/formatted/train_info.csv',
                  '--val-info-fp=/home/jupyter/sound_event_detection/datasets/powdermill/formatted/val_info.csv',
                  '--test-info-fp=/home/jupyter/sound_event_detection/datasets/powdermill/formatted/test_info.csv',
                  '--project-dir=/home/jupyter/sound_event_detection/projects/powdermill_experiment'] & FG
  
  params_file = '/home/jupyter/sound_event_detection/projects/powdermill_experiment/project_config.yaml'
  with open(params_file, 'r') as f:
    cfg = yaml.safe_load(f)
    
  new_label_mapping = {}
  
  for k in cfg['label_mapping']:
    if k == 'Unknown':
      new_label_mapping[k] = 'Unknown'
    # omit NOCA, AMCR, BLJA because they have very long boxes
    elif k in ['EATO','WOTH','BCCH','BTNW','TUTI','REVI','OVEN','COYE','BGGN','SCTA']:
      new_label_mapping[k] = k
      
  cfg['label_mapping'] = new_label_mapping
  cfg['label_set'] = ['EATO','WOTH','BCCH','BTNW','TUTI','REVI','OVEN','COYE','BGGN','SCTA']
  
  with open(params_file, "w") as f:
    yaml.dump(cfg, f)

  j = 0
  for lr in [.0001, .00005, .00001]:
    for batch_size in [32, 4]:
      local['python']['../main.py',
                      'train-model',
                      '--project-config-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment/project_config.yaml',
                      f'--name=m{j}',
                      '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                      '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                      f'--lr={lr}',
                      f'--batch-size={batch_size}'
                     ] & FG
      j += 1

if __name__ == "__main__":
  main()