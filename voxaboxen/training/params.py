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
  parser.add_argument('--is-test', '-t', action='store_true', help='run a quick version for testing')
  parser.add_argument('--overwrite', action='store_true', help='overwrite an experiment of the same name, if it exists')

  # Data
  parser.add_argument('--project-config-fp', type = str, required=True)
  parser.add_argument('--clip-duration', type=float, default=6.0, help = "clip duration, in seconds")
  parser.add_argument('--clip-hop', type=float, default=None, help = "clip hop, in seconds. If None, automatically set to be half clip duration. Used only during training; clip hop is automatically set to be 1/2 clip duration for inference")
  parser.add_argument('--train-info-fp', type=str, required=False, help = "train info, to override project train info")
  parser.add_argument('--num-workers', type=int, default=8)

  # Model
  parser.add_argument('--bidirectional', action='store_true', help="train and inference in both directions and combine results")
  parser.add_argument('--sr', type=int, default=16000)
  parser.add_argument('--scale-factor', type=int, default = 320, help = "downscaling performed by encoder")
  parser.add_argument('--encoder-type', type=str, default = "aves", choices = ["aves", "hubert_base", "frame_atst", "beats"])
  parser.add_argument('--prediction-scale-factor', type=int, default = 1, help = "downsampling rate from encoder sr to prediction sr. Deprecated.")
  parser.add_argument('--detection-threshold', type=float, default = 0.5, help = "output probability to count as positive detection")
  parser.add_argument('--rms-norm', action="store_true", help = "If true, apply rms normalization to each clip")
  parser.add_argument('--previous-checkpoint-fp', type=str, default=None, help="path to checkpoint of previously trained detection model")
  
  parser.add_argument('--stereo', action='store_true', help="If passed, will process stereo data as stereo. order of channels matters")
  parser.add_argument('--multichannel', action='store_true', help="If passed, will encode each audio channel seperately, then add together the encoded audio before final layer")
  parser.add_argument('--segmentation-based', action='store_true', help="If passed, will make predictions based on frame-wise segmentations rather than box starts")
  parser.add_argument('--comb-discard-thresh', type=float, default=0.75, help="If bidirectional, sets threshold for combining forward and backward predictions")
  parser.add_argument('--comb-iou-threshold', type=float, default=0.5, help="minimum iou to match a forward and backward prediction")
  # parser.add_argument('--reload-from', type=str)
  
  # Encoder-specific
  ## AVES
  parser.add_argument('--aves-config-fp', type=str, default = "weights/birdaves-biox-base.torchaudio.model_config.json")
  parser.add_argument('--aves-url', type=str, default = "https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-base.torchaudio.pt")
  ## Frame-ATST
  parser.add_argument('--frame-atst-weight-fp', type=str, default = "weights/atstframe_base.ckpt")
  ## BEATs
  parser.add_argument('--beats-checkpoint-fp', type=str, default = "weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
  
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
  parser.add_argument('--delete-short-dur-sec', type=float, default=0.1, help="if using segmentation based model, delete vox shorter than this as a post-processing step")
  parser.add_argument('--fill-holes-dur-sec', type=float, default=0.1, help="if using segmentation based model, fill holes shorter than this as a post-processing step")
  
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
  assert ((args.clip_duration * args.sr)/(2*args.scale_factor)).is_integer(), "Must pick clip duration to ensure no rounding errors during inference"
  if args.segmentation_based and (args.rho!=1):
      import warnings
      warnings.warn("when using segmentation-based framework, recommend setting args.rho=1")
  if args.bidirectional and args.segmentation_based:
      raise ValueError("bidirectional and segmentation settings are not currently compatible")
  if args.encoder_type == "aves":
      assert args.scale_factor == 320, "AVES requires scale-factor == 320"
  elif args.encoder_type == "hubert_base":
      assert args.scale_factor == 320, "hubert_base requires scale-factor == 320"
  elif args.encoder_type == "frame_atst":
      assert args.scale_factor == 640, "frame_atst requires scale-factor == 640"
      assert args.clip_duration == 10, "frame_atst expects clip duration of 10 seconds"
  elif args.encoder_type == "beats":
      assert args.scale_factor == 320, "beats requires scale-factor == 320"
  else:
      import warnings
      warnings.warn("Did not confirm correct scale factor for chosen encoder type")