import argparse
import torch
import numpy as np
import random
import logging
import os
import yaml

def parse_args(args):
  parser = argparse.ArgumentParser()
  
  # General
  parser.add_argument('--output-dir', type=str, default="/home/jupyter/sound_event_detection/logs")
  parser.add_argument('--name', type = str, required=True)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--annotation-selection-tables-dir', type = str, default = '/home/jupyter/carrion_crows_data/Annotations_revised_by_Daniela.cleaned/selection_tables')

  # Data
  parser.add_argument('--label-mapping-config-fp', type = str, default = "/home/jupyter/sound_event_detection/config/focal_nonfocal.yaml")
  parser.add_argument('--clip-duration', type=float, default=20.0, help = "clip duration, in seconds")
  parser.add_argument('--clip-hop', type=float, default=10.0, help = "clip hop, in seconds")
  parser.add_argument('--dev-info-fp', type=str, default = "/home/jupyter/carrion_crows_data/call_detection_data.revised_anno.all/dev_info.csv")
  parser.add_argument('--test-info-fp', type=str, default = "/home/jupyter/carrion_crows_data/call_detection_data.revised_anno.all/test_info.csv")
  parser.add_argument('--num-workers', type=int, default=8)
  parser.add_argument('--num-files-val', type=int, default=1)
  
  # Model
  parser.add_argument('--sr', type=int, default=16000)
  parser.add_argument('--scale-factor', type=int, default = 320, help = "downscaling performed by aves")
  parser.add_argument('--model-weight-fp', type=str, default = "/home/jupyter/carrion_crows/clip/pretrained_weights/aves-base-bio.pt")
  parser.add_argument('--prediction-scale-factor', type=int, default = 1, help = "downsampling rate from aves sr to prediction sr")
  parser.add_argument('--detection-threshold', type=float, default = 0.5, help = "output probability to count as positive detection")
  parser.add_argument('--rms-norm', action="store_true", help = "If true, apply rms normalization to each clip")
  
  # Training
  parser.add_argument('--batch-size', type=int, required=True) 
  parser.add_argument('--lr', type=float, required=True) 
  parser.add_argument('--n-epochs', type=int, required=True)
  parser.add_argument('--unfreeze-encoder-epoch', type=int, default=1)
  parser.add_argument('--end-mask-perc', type=float, default = 0.1, help="During training, mask loss from a percentage of the final frames") 
  parser.add_argument('--omit-empty-clip-prob', type=float, default=0.95, help="if a clip has no annotations, do not use for training with this probability")
  parser.add_argument('--lamb', type=float, default=.003, help="parameter controlling strength regression loss")                 
  
  # Augmentations
  parser.add_argument('--amp-aug', action ="store_true", help="Whether to use amplitude augmentation") 
  parser.add_argument('--amp-aug-low-r', type=float, default = 0.8) 
  parser.add_argument('--amp-aug-high-r', type=float, default = 1.2) 
  parser.add_argument('--mixup', action ="store_true", help="Whether to use mixup augmentation") 
  
  args = parser.parse_args()
  # csv_attrs = ["label_set"]
  # for attr in csv_attrs:
  #     if isinstance(getattr(args, attr), str):
  #         setattr(args, attr, getattr(args, attr).split(","))
  
  args = read_config(args)
  
  return args

def read_config(args):
  with open(args.label_mapping_config_fp, 'r') as f:
    label_mapping_config = yaml.safe_load(f)
    
  for key in label_mapping_config:
    setattr(args,key,label_mapping_config[key])
    
  return args

def set_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

def save_params(args):
  """ Save a copy of the params used for this experiment """
  logging.info("Params:")
  params_file = os.path.join(args.experiment_dir, "params.yaml")

  args_dict = {}
  for name in sorted(vars(args)):
    val = getattr(args, name)
    logging.info(f"  {name}: {val}")
    args_dict[name] = val

  with open(params_file, "w") as f:
    yaml.dump(args_dict, f)

  # for name in sorted(vars(args)):
  #     val = getattr(args, name)
  #     logging.info(f"  {name}: {val}")
  #     f.write(f"{name}: {val}\n")
            
      
def load_params(fp):
  with open(fp, 'r') as f:
    args_dict = yaml.safe_load(f)

  args = argparse.Namespace()

  for key in args_dict:
    setattr(args, key, args_dict[key])

  return args