from plumbum import local, FG
import os
import yaml
import numpy as np


def main():
  # Project setup
  local['python']['../main.py',
                  'project-setup',
                  '--train-pool-info-fp=/home/jupyter/sound_event_detection/datasets/birdvox_full_night/formatted/train_pool_info.csv',
                  '--test-info-fp=/home/jupyter/sound_event_detection/datasets/birdvox_full_night/formatted/test_info.csv',
                  '--project-dir=/home/jupyter/sound_event_detection/projects/bvfn_experiment_random'] & FG
  
  for i in range(20):  
    # random control
    for j in range(5):
      local['python']['../main.py',
                      'active-learning-sampling',
                      '--project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_experiment_random/project_config.yaml',
                      '--sampling-method=random',
                      f'--random-clips-per-file={(i+1)*10}', 
                      '--sample-duration=10', 
                      f'--sequence-name=control_{i}_{j}',
                      '--query-oracle',
                      f'--seed={j}',
                      f'--max-n-clips-to-sample={(i+1)*60}'] & FG

      local['python']['../main.py',
                      'train-model',
                      '--project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_experiment_random/project_config.yaml',
                      f'--name=m_{i}_{j}',
                      '--clip-duration=2',
                      '--clip-hop=1',
                      '--omit-empty-clip-prob=0',
                      f'--seed={j}',
                      f'--train-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_experiment_random/active_learning/train_info_control_{i}_{j}_0.csv'
                     ] & FG

if __name__ == "__main__":
  main()
