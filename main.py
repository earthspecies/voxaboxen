# New workflow, under main:
# 1. `project_setup`
# 2. `active_learning_sampling`
# 3. annotate or `query_oracle`
# 4. `trian_model`
# 5. Repeat 2-4 as desired.

# These are set up in `benchmarking`, in scripts

import sys

def main(mode, args):  
  if mode == 'project_setup':
    import source.project.project_setup.project_setup as project_setup
    project_setup(args)
  
  if mode == 'active_learning_sampling':
    pass
  
  if mode == 'query_oracle':
    pass
  
  if mode == 'train_model':
    import source.training.train_model.train_model as train_model
    train_model(args)

if __name__ == "__main__":
  ins = sys.argv
  mode = ins[1]
  args = ins[2:]
  main(mode, args)