import os
import yaml
import argparse

def save_params(args):
  """ Save a copy of the params used for this experiment """
  params_file = os.path.join(args.project_dir, "project_config.yaml")

  args_dict = {}
  for name in sorted(vars(args)):
    val = getattr(args, name)
    args_dict[name] = val

  with open(params_file, "w") as f:
    yaml.dump(args_dict, f)
    
  print(f"Saved config to {params_file}. You may now edit this file if you want some classes to be omitted or treated as Unknown")
  
def parse_project_args(args):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--train-info-fp', type=str, required=True, help = "filepath of csv with train info")
  parser.add_argument('--val-info-fp', type=str, default=None, help = "filepath of csv with val info")
  parser.add_argument('--test-info-fp', type=str, required=True, help = "filepath of csv with test info")
  parser.add_argument('--project-dir', type=str, required=True, help = "directory where project will be stored")
  
  args = parser.parse_args(args)  
  return args