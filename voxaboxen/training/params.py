import argparse
import torch
import numpy as np
import random
import logging
import os
import yaml

def parse_args(args,allow_unknown=False):
  parser = argparse.ArgumentParser()
  
  # General
  parser.add_argument('--name', type = str, required=True)
  parser.add_argument('--seed', type=int, default=0)

  # Data
  parser.add_argument('--project-config-fp', type = str, required=True)
  parser.add_argument('--clip-duration', type=float, default=6.0, help = "clip duration, in seconds")
  parser.add_argument('--clip-hop', type=float, default=None, help = "clip hop, in seconds. If None, automatically set to be half clip duration. Used only during training; clip hop is automatically set to be 1/2 clip duration for inference")
  parser.add_argument('--train-info-fp', type=str, required=False, help = "train info, to override project train info")
  parser.add_argument('--num-workers', type=int, default=8)
  
  # Model
  parser.add_argument('--sr', type=int, default=16000)
  parser.add_argument('--scale-factor', type=int, default = 320, help = "downscaling performed by aves")
  parser.add_argument('--aves-config-fp', type=str, default = "weights/birdaves-biox-base.torchaudio.model_config.json")
  parser.add_argument('--prediction-scale-factor', type=int, default = 1, help = "downsampling rate from aves sr to prediction sr")
  parser.add_argument('--detection-threshold', type=float, default = 0.5, help = "output probability to count as positive detection")
  parser.add_argument('--rms-norm', action="store_true", help = "If true, apply rms normalization to each clip")
  parser.add_argument('--previous-checkpoint-fp', type=str, default=None, help="path to checkpoint of previously trained detection model")
  parser.add_argument('--aves-url', type=str, default = "https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-base.torchaudio.pt")
  parser.add_argument('--stereo', action='store_true', help="If passed, will process stereo data as stereo. order of channels matters")
  parser.add_argument('--multichannel', action='store_true', help="If passed, will encode each audio channel seperately, then add together the encoded audio before final layer")
  parser.add_argument('--segmentation-based', action='store_true', help="If passed, will make predictions based on frame-wise segmentations rather than box starts")

  # Training
  parser.add_argument('--batch-size', type=int, default=32) 
  parser.add_argument('--lr', type=float, default=.00005) 
  parser.add_argument('--n-epochs', type=int, default=50)
  parser.add_argument('--unfreeze-encoder-epoch', type=int, default=3)
  parser.add_argument('--end-mask-perc', type=float, default = 0.1, help="During training, mask loss from a percentage of the frames on each end of the clip") 
  parser.add_argument('--omit-empty-clip-prob', type=float, default=0, help="if a clip has no annotations, do not use for training with this probability")
  parser.add_argument('--lamb', type=float, default=.04, help="parameter controlling strength regression loss")
  parser.add_argument('--rho', type=float, default = .01, help="parameter controlling strength of classification loss")
  # parser.add_argument('--step-size', type=int, default=7, help="number epochs between lr decrease")
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
  parser.add_argument('--nms', type = str, default='soft_nms', choices = ['none', 'nms', 'soft_nms'], help="Whether to apply additional nms after finding peaks")
  parser.add_argument('--soft-nms-sigma', type = float, default = 0.5)
  parser.add_argument('--soft-nms-thresh', type = float, default = 0.001)
  parser.add_argument('--nms-thresh', type = float, default = 0.5)
  
  if allow_unknown:
    args, remaining = parser.parse_known_args(args)
  else:
    args = parser.parse_args(args)
  
  args = read_config(args)
  check_config(args)

  if args.clip_hop is None:
    setattr(args, "clip_hop", args.clip_duration/2)
  
  if allow_unknown:
    return args, remaining
  else:
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
  assert ((args.clip_duration * args.sr)/(4*args.scale_factor)).is_integer(), "Must pick clip duration to ensure no rounding errors during inference"