# Workflow, under main:
# 1. `project-setup`
# 2. `train-model`
# 3. 'inference'

import sys

def main(mode, args):  
  
  if mode == 'project-setup':
    from source.project.project_setup import project_setup
    project_setup(args)
  
  if mode == 'train-model':
    from source.training.train_model import train_model
    train_model(args)
  
  if mode == 'train-comparison':
    from source.comparisons.train import train
    train(args)
    
  if mode == 'inference':
    from source.inference.inference import inference
    inference(args)

if __name__ == "__main__":
  ins = sys.argv
  mode = ins[1]
  args = ins[2:]
  main(mode, args)