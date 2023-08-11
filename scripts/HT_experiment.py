from plumbum import local, FG
import os
import yaml
import numpy as np


def main():
  # Project setup
  local['python']['../main.py',
                  'project-setup',
                  '--train-info-fp=/home/jupyter/sound_event_detection/datasets/HT/formatted/train_info.csv',
                  '--val-info-fp=/home/jupyter/sound_event_detection/datasets/HT/formatted/val_info.csv',
                  '--test-info-fp=/home/jupyter/sound_event_detection/datasets/HT/formatted/test_info.csv',
                  '--project-dir=/home/jupyter/sound_event_detection/projects/HT_experiment'] & FG
  
  j = 0
  for lr in [.0001, .00005, .00001]:
    for batch_size in [32, 4]:
      local['python']['../main.py',
                      'train-model',
                      '--project-config-fp=/home/jupyter/sound_event_detection/projects/HT_experiment/project_config.yaml',
                      f'--name=m{j}',
                      '--aves-config-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.model_config.json',
                      '--aves-model-weight-fp=/home/jupyter/sound_event_detection/weights/aves-base-bio.torchaudio.pt',
                      f'--lr={lr}',
                      f'--batch-size={batch_size}'
                     ] & FG
      j += 1

if __name__ == "__main__":
  main()
