# New workflow, under main:
# 1. `project-setup`
# 2. `active-learning-sampling`
# 3. annotate
# 4. `trian-model`
# 5. Repeat 2-4 as desired.

# These are set up in `benchmarking`, in scripts

# python main.py train-model --output-dir=/home/jupyter/sound_event_detection/projects/bvfn_debug --name=d0 --label-mapping-config-fp=/home/jupyter/sound_event_detection/projects/bvfn_debug/project_config.yaml --clip-duration=4 --clip-hop=2 --train-info-fp=/home/jupyter/sound_event_detection/datasets/birdvox_full_night/formatted/train_pool_info.csv --batch-size=32 --lr=0.0001 --n-epochs=4 --unfreeze-encoder-epoch=1 --omit-empty-clip-prob=1 --step-size=2

import sys
from source.project.project_setup import project_setup


def main(mode, args):  
  
  if mode == 'project-setup':
    project_setup(args)
  
  if mode == 'active-learning-sampling':
    pass
  
  if mode == 'train-model':
    from source.training.train_model import train_model
    train_model(args)

if __name__ == "__main__":
  ins = sys.argv
  mode = ins[1]
  args = ins[2:]
  main(mode, args)