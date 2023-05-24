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
                  '--project-dir=/home/jupyter/sound_event_detection/projects/bvfn_experiment_uncertainty'] & FG
  
  for i in range(20):
    # uncertainty sampling with random init
    if i == 0 :
      local['python']['../main.py',
                      'active-learning-sampling',
                      '--project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_experiment_uncertainty/project_config.yaml',
                      '--sampling-method=coreset',
                      '--sample-duration=10', 
                      '--sequence-name=unc',
                      '--query-oracle',
                      '--max-n-clips-to-sample=60'] & FG
      
    elif i<3 :
      local['python']['../main.py',
                      'active-learning-sampling',
                      '--project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_experiment_uncertainty/project_config.yaml',
                      '--sampling-method=coreset',
                      '--sample-duration=10', 
                      f'--prev-iteration-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_experiment_uncertainty/active_learning/train_info_unc_{i-1}.csv',
                      f'--model-args-fp=/home/jupyter/sound_event_detection/projects/bvfn_experiment_uncertainty/m{i-1}/params.yaml',
                      '--query-oracle',
                      '--max-n-clips-to-sample=60'] & FG
      
    else:
      local['python']['../main.py',
                      'active-learning-sampling',
                      '--project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_experiment_uncertainty/project_config.yaml',
                      '--sampling-method=uncertainty',
                      '--uncertainty-clips-per-file=10', 
                      '--sample-duration=10', 
                      f'--prev-iteration-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_experiment_uncertainty/active_learning/train_info_unc_{i-1}.csv',
                      f'--model-args-fp=/home/jupyter/sound_event_detection/projects/bvfn_experiment_uncertainty/m{i-1}/params.yaml',
                      '--query-oracle',
                      '--max-n-clips-to-sample=60'] & FG
      
    local['python']['../main.py',
                    'train-model',
                    '--project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_experiment_uncertainty/project_config.yaml',
                    f'--name=m{i}',
                    '--clip-duration=2',
                    '--clip-hop=1',
                    '--omit-empty-clip-prob=0',
                    f'--train-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_experiment_uncertainty/active_learning/train_info_unc_{i}.csv'
                   ] & FG       

if __name__ == "__main__":
  main()
