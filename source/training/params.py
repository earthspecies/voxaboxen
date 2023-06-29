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
  parser.add_argument('--name', type = str, required=True)
  parser.add_argument('--seed', type=int, default=0)

  # Data
  parser.add_argument('--project-config-fp', type = str, required=True)
  parser.add_argument('--clip-duration', type=float, default=20.0, help = "clip duration, in seconds")
  parser.add_argument('--clip-hop', type=float, default=10.0, help = "clip hop, in seconds")
  parser.add_argument('--train-info-fp', type=str, required=True)
  parser.add_argument('--num-workers', type=int, default=8)
  
  # Model
  parser.add_argument('--sr', type=int, default=16000)
  parser.add_argument('--scale-factor', type=int, default = 320, help = "downscaling performed by aves")
  parser.add_argument('--aves-model-weight-fp', type=str, default = "/home/jupyter/carrion_crows/clip/pretrained_weights/aves-base-bio.pt")
  parser.add_argument('--prediction-scale-factor', type=int, default = 1, help = "downsampling rate from aves sr to prediction sr")
  parser.add_argument('--detection-threshold', type=float, default = 0.5, help = "output probability to count as positive detection")
  parser.add_argument('--rms-norm', action="store_true", help = "If true, apply rms normalization to each clip")
  parser.add_argument('--previous-checkpoint-fp', type=str, default=None, help="path to checkpoint of previously trained detection model")
  
  # Training
  parser.add_argument('--batch-size', type=int, default=32) 
  parser.add_argument('--lr', type=float, default=.0005) 
  parser.add_argument('--n-epochs', type=int, default=28)
  parser.add_argument('--unfreeze-encoder-epoch', type=int, default=7)
  parser.add_argument('--end-mask-perc', type=float, default = 0.1, help="During training, mask loss from a percentage of the frames on each end of the clip") 
  parser.add_argument('--omit-empty-clip-prob', type=float, default=0.5, help="if a clip has no annotations, do not use for training with this probability")
  parser.add_argument('--lamb', type=float, default=.04, help="parameter controlling strength regression loss")
  parser.add_argument('--rho', type=float, default = .01, help="parameter controlling strength of classification loss")
  parser.add_argument('--step-size', type=int, default=7, help="number epochs between lr decrease")
  parser.add_argument('--model-selection-iou', type=float, default=0.5, help="iou for used for computing f1 for early stopping")
  parser.add_argument('--model-selection-class-threshold', type=float, default=0.5, help="class threshold for used for computing f1 for early stopping")

  parser.add_argument('--early-stopping', action ="store_true", help="Whether to use early stopping based on val performance")
  parser.add_argument('--pos-loss-weight', type=float, default=1, help="Weights positive component of loss")
  
  # Augmentations
  parser.add_argument('--amp-aug', action ="store_true", help="Whether to use amplitude augmentation") 
  parser.add_argument('--amp-aug-low-r', type=float, default = 0.8) 
  parser.add_argument('--amp-aug-high-r', type=float, default = 1.2) 
  parser.add_argument('--mixup', action ="store_true", help="Whether to use mixup augmentation") 
  
  # Inference
  parser.add_argument('--peak-distance', type=float, default=5, help="for finding peaks in detection probability, what radius to use for detecting local maxima. In output frame rate.")
  
  args = parser.parse_args(args)
  args = read_config(args)
  
  check_config(args)
  
  return args

def read_config(args):
  with open(args.project_config_fp, 'r') as f:
    project_config = yaml.safe_load(f)
    
  for key in project_config:
    setattr(args,key,project_config[key])
    
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
      
def load_params(fp):
  with open(fp, 'r') as f:
    args_dict = yaml.safe_load(f)

  args = argparse.Namespace()

  for key in args_dict:
    setattr(args, key, args_dict[key])

  return args

def check_config(args):
  assert args.end_mask_perc < 0.25, "Masking above 25% of each end during training will interfere with inference"