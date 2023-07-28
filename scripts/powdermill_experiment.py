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

  local['python']['../main.py',
                  'train-model',
                  '--project-config-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment/project_config.yaml',
                  '--name=m0',
                  '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                  '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                 ] & FG
  
  local['python']['../main.py',
                  'train-model',
                  '--project-config-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment/project_config.yaml',
                  '--name=m1',
                  '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                  '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                  '--lr=.00008'
                 ] & FG 
  
  local['python']['../main.py',
                  'train-model',
                  '--project-config-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment/project_config.yaml',
                  '--name=m2',
                  '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                  '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                  '--lamb=.08'
                 ] & FG 
  
  local['python']['../main.py',
                  'train-model',
                  '--project-config-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment/project_config.yaml',
                  '--name=m3',
                  '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                  '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                  '--rho=.02'
                 ] & FG 
  
  local['python']['../main.py',
                  'train-model',
                  '--project-config-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment/project_config.yaml',
                  '--name=m4',
                  '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                  '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                  '--pos-loss-weight=2'
                 ] & FG 
  
  local['python']['../main.py',
                  'train-model',
                  '--project-config-fp=/home/jupyter/sound_event_detection/projects/powdermill_experiment/project_config.yaml',
                  '--name=m5',
                  '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                  '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                  '--batch-size=8'
                 ] & FG 
  
  # Print results
  
  evals = {}
  for i in range(6):
    name = f'm{i}'
    results_fp = f'/home/jupyter/sound_event_detection/projects/powdermill_experiment/{name}/val_results/metrics_iou_0.5_class_threshold_0.5.yaml'
    with open(results_fp, 'r') as f:
      results = yaml.safe_load(f)
    evals[name] = results['macro']['f1']
    
  print(evals)


if __name__ == "__main__":
  main()