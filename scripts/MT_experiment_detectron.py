from plumbum import local, FG
import os
import yaml
import numpy as np


def main():
  # Project setup
  local['python']['../main.py',
                  'project-setup',
                  '--train-info-fp=/home/jupyter/sound_event_detection/datasets/MT/formatted/train_info.csv',
                  '--val-info-fp=/home/jupyter/sound_event_detection/datasets/MT/formatted/val_info.csv',
                  '--test-info-fp=/home/jupyter/sound_event_detection/datasets/MT/formatted/test_info.csv',
                  '--project-dir=/home/jupyter/sound_event_detection/projects/MT_experiment_detectron'] & FG
  
  j = 0
  for lr in [.005, .001, .0005]:
    for batch_size in [16, 4]:
      try:
        local['python']['../main.py',
                        'train-comparison',
                        '--project-config-fp=/home/jupyter/sound_event_detection/projects/MT_experiment_detectron/project_config.yaml',
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
