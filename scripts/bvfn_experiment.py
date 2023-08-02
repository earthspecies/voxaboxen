from plumbum import local, FG
import os
import yaml
import numpy as np


def main():
  # Project setup
  local['python']['../main.py',
                  'project-setup',
                  '--train-info-fp=/home/jupyter/sound_event_detection/datasets/birdvox_full_night/formatted/train_info.csv',
                  '--val-info-fp=/home/jupyter/sound_event_detection/datasets/birdvox_full_night/formatted/val_info.csv',
                  '--test-info-fp=/home/jupyter/sound_event_detection/datasets/birdvox_full_night/formatted/test_info.csv',
                  '--project-dir=/home/jupyter/sound_event_detection/projects/birdvox_full_night_experiment'] & FG
  
  local['python']['../main.py',
                    'train-model',
                    '--project-config-fp=/home/jupyter/sound_event_detection/projects/birdvox_full_night_experiment/project_config.yaml',
                    '--name=m0',
                    '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                    '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                    '--clip-duration=2.0',
                    '--clip-hop=1.0',
                    '--omit-empty-clip-prob=0.75'
                   ] & FG
  
  local['python']['../main.py',
                  'train-model',
                  '--project-config-fp=/home/jupyter/sound_event_detection/projects/birdvox_full_night_experiment/project_config.yaml',
                  '--name=m1',
                  '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                  '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                  '--clip-duration=2.0',
                  '--clip-hop=1.0',
                  '--omit-empty-clip-prob=0.75',
                  '--lr=.00008'
                 ] & FG 
  
  local['python']['../main.py',
                  'train-model',
                  '--project-config-fp=/home/jupyter/sound_event_detection/projects/birdvox_full_night_experiment/project_config.yaml',
                  '--name=m2',
                  '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                  '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                  '--clip-duration=2.0',
                  '--clip-hop=1.0',
                  '--omit-empty-clip-prob=0.75',
                  '--pos-loss-weight=2'
                 ] & FG 
  
  local['python']['../main.py',
                  'train-model',
                  '--project-config-fp=/home/jupyter/sound_event_detection/projects/birdvox_full_night_experiment/project_config.yaml',
                  '--name=m3',
                  '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                  '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                  '--clip-duration=2.0',
                  '--clip-hop=1.0',
                  '--omit-empty-clip-prob=0.75',
                  '--batch-size=8'
                 ] & FG 
  
  # Print results
  
  evals = {}
  for i in range(1):
    name = f'm{i}'
    results_fp = f'/home/jupyter/sound_event_detection/projects/birdvox_full_night_experiment/{name}/val_results/metrics_iou_0.5_class_threshold_0.5.yaml'
    with open(results_fp, 'r') as f:
      results = yaml.safe_load(f)
    evals[name] = results['macro']['f1']
    
  print(evals)

if __name__ == "__main__":
  main()
