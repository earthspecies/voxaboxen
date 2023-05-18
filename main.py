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

# python main.py project-setup --train-pool-info-fp=/home/jupyter/sound_event_detection/datasets/alala/formatted/train_pool_info.csv --test-info-fp=/home/jupyter/sound_event_detection/datasets/alala/formatted/test_info.csv --val-info-fp=/home/jupyter/sound_event_detection/datasets/alala/formatted/val_info.csv --project-dir=/home/jupyter/sound_event_detection/projects/alala


# python main.py train-model --name=debug --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --clip-duration=2 --clip-hop=1 --train-info-fp=/home/jupyter/sound_event_detection/datasets/birdvox_full_night/formatted/train_pool_info.csv --n-epochs=1 --omit-empty-clip-prob=1

# python main.py train-model --name=debug --project-config-fp=/home/jupyter/sound_event_detection/projects/alala/project_config.yaml --clip-duration=6 --clip-hop=3 --train-info-fp=/home/jupyter/sound_event_detection/datasets/alala/formatted/train_info_iter_0.csv --early-stopping

# python main.py active-learning-sampling --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --uncertainty-clips-per-file=10 --sample-duration=10 --model-args-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/debug/params.yaml


# python main.py active-learning-sampling --sampling-method=random --project-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --sample-duration=5


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