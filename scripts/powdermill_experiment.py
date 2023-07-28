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
    elif k in ['EATO','WOTH','BCCH','BTNW','TUTI','NOCA','REVI','AMCR','BLJA','OVEN']:
      new_label_mapping[k] = k
      
  cfg['label_mapping'] = new_label_mapping
  cfg['label_set'] = ['EATO','WOTH','BCCH','BTNW','TUTI','NOCA','REVI','AMCR','BLJA','OVEN']
  
  with open(params_file, "w") as f:
    yaml.dump(cfg, f)

  local['python']['../main.py',
                  'train-model',
                  '--project-config-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment/project_config.yaml',
                  '--name=m0',
                  '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                  '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                 ] & FG   

if __name__ == "__main__":
  main()