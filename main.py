# Workflow, under main:
# 1. `project-setup`
# 1.b edit project config to reflect label mapping preferences
# 2. `active-learning-sampling`
# 3. annotate
# 4. `trian-model`
# 5. Repeat 2-4 as desired.
# 6. 'inference'

# These are set up in the different `experiment` files, in scripts

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
    from source.inference.inference import inference
    inference(args)

if __name__ == "__main__":
  ins = sys.argv
  mode = ins[1]
  args = ins[2:]
  main(mode, args)