import os
import yaml
import sys
import pandas as pd
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
    
  print(f"Saved config to {params_file}. You may now edit this file if you want some classes to be treated as Unknown")
  
def parse_args(args):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--train-info-fp', type=str, required=True, help = "filepath of csv with train pool info")
  parser.add_argument('--val-info-fp', type=str, default=None, help = "filepath of csv with val pool info")
  parser.add_argument('--test-info-fp', type=str, required=True, help = "filepath of csv with test pool info")
  parser.add_argument('--project-dir', type=str, required=True, help = "directory where project will be stored")
  
  al_args = parser.parse_args()  
  return al_args

def project_setup(args):
  args = parse_args(args)
  
  annots = []
  for info_fp in [args.train_info_fp, args.val_info_fp, args.test_info_fp]:
    if info_fp is None:
      continue
    
    info = pd.read_csv(info_fp)
    annot_fps = list(info['selection_table_fp'])
    
    for annot_fp in annot_fps:
      if os.path.exists(annot_fp):
        selection_table = pd.read_csv(annot_fp, delimiter = '\t')
        annots = list(selection_table['Annotation'])
        annots.extend(annots)
        
  label_set = sorted(set(annots))
  label_mapping = {x : x for x in label_set}
  unknown_label = 'Unknown'
  
  setattr(args, "label_set", label_set)
  setattr(args, "label_mapping", label_mapping)
  setattr(args, "unknown_label", unknown_label)
  
  save_params(args)

if __name__ == "__main__":
  project_setup(sys.argv[1:])