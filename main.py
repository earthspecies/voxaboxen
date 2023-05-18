# New workflow, under main:
# 1. `project-setup`
# 1.b edit project config to reflect label mapping preferences
# 2. `active-learning-sampling`
# 3. annotate
# 4. `trian-model`
# 5. Repeat 2-4 as desired.
# 6. 'inference'

# These are set up in `benchmarking`, in scripts

# python main.py project-setup --train-pool-info-fp=/home/jupyter/sound_event_detection/datasets/birdvox_full_night/formatted/train_pool_info.csv --test-info-fp=/home/jupyter/sound_event_detection/datasets/birdvox_full_night/formatted/test_info.csv --project-dir=/home/jupyter/sound_event_detection/projects/bvfn_debug

# python main.py active-learning-sampling --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --sampling-method=random --random-clips-per-file=10 --sample-duration=10 --model-args-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/m0/params.yaml --sequence-name=debug --query-oracle

# python main.py train-model --name=m0 --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --clip-duration=2 --clip-hop=1 --train-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/active_learning/train_info_debug_0.csv

# python main.py active-learning-sampling --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --uncertainty-clips-per-file=10 --sample-duration=10 --model-args-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/m0/params.yaml --prev-iteration-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/active_learning/train_info_debug_0.csv --query-oracle

# python main.py train-model --name=m1 --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --clip-duration=2 --clip-hop=1 --train-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/active_learning/train_info_debug_1.csv

# python main.py active-learning-sampling --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --uncertainty-clips-per-file=10 --sample-duration=10 --model-args-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/m1/params.yaml --prev-iteration-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/active_learning/train_info_debug_1.csv --query-oracle

# python main.py train-model --name=m2 --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --clip-duration=2 --clip-hop=1 --train-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/active_learning/train_info_debug_2.csv

# python main.py active-learning-sampling --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --uncertainty-clips-per-file=10 --sample-duration=10 --model-args-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/m2/params.yaml --prev-iteration-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/active_learning/train_info_debug_2.csv --query-oracle

# python main.py train-model --name=m3 --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --clip-duration=2 --clip-hop=1 --train-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/active_learning/train_info_debug_3.csv

# python main.py active-learning-sampling --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --uncertainty-clips-per-file=10 --sample-duration=10 --model-args-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/m2/params.yaml --prev-iteration-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/active_learning/train_info_debug_2.csv --query-oracle

# python main.py train-model --name=m3 --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --clip-duration=2 --clip-hop=1 --train-info-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/active_learning/train_info_debug_3.csv





# python main.py project-setup --train-pool-info-fp=/home/jupyter/sound_event_detection/datasets/alala/formatted/train_pool_info.csv --test-info-fp=/home/jupyter/sound_event_detection/datasets/alala/formatted/test_info.csv --val-info-fp=/home/jupyter/sound_event_detection/datasets/alala/formatted/val_info.csv --project-dir=/home/jupyter/sound_event_detection/projects/alala

# python main.py train-model --name=debug --project-config-fp=/home/jupyter/sound_event_detection/projects/alala/project_config.yaml --clip-duration=6 --clip-hop=3 --train-info-fp=/home/jupyter/sound_event_detection/datasets/alala/formatted/train_info_iter_0.csv --early-stopping

import sys

def main(mode, args):  
  
  if mode == 'project-setup':
    from source.project.project_setup import project_setup
    project_setup(args)
  
  if mode == 'active-learning-sampling':
    from source.active_learning.active_learning_sampling import active_learning_sampling
    active_learning_sampling(args)
  
  if mode == 'train-model':
    from source.training.train_model import train_model
    train_model(args)
    
  if mode == 'inference':
    pass

if __name__ == "__main__":
  ins = sys.argv
  mode = ins[1]
  args = ins[2:]
  main(mode, args)