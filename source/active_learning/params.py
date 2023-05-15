import argparse
import os
import yaml

def parse_al_args(al_args):
  parser = argparse.ArgumentParser()
  
  # General
  parser.add_argument('--model-args-fp', type=str, required=True, help = "filepath of model params saved as a yaml")
  parser.add_argument('--model-checkpoint-fp', type=str, required=True, help = "filepath of model checkpoint")
  parser.add_argument('--seed', type=int, default=0, help="random seed")
  parser.add_argument('--name', type=str, required=True)
  parser.add_argument('--output-dir', type=str, default=None)
  # parser.add_argument('--candidate-manifest-fp', required=True, help='fp to tsv list of wav files to sample')
  parser.add_argument('--sample-duration', type = float, default = 30, help = 'duration of clip to sample for annotation, in seconds')
  parser.add_argument('--max-n-clips-to-sample', type = int, default = 120, help = 'how many clips to sample')
  parser.add_argument('--clips-per-file', type = int, default = 1, help = 'how many clips to sample per file')
  parser.add_argument('--uncertainty-discount-factor', type = float, default = 0.8, help = 'geometric weighting of uncertainties. Uncertainties are sorted and then the lower ranking uncertainties count for less. Closer to 0 discourages sampling clips with lots of detections')
  parser.add_argument('--uncertainty-detection-threshold', type=float, default = 0.1, help = 'ignore detection peaks lower than this value, for the purpose of computing uncertainty')
  al_args = parser.parse_args(al_args)  
  return al_args

def save_params(al_args):
  """ Save a copy of the params used for this experiment """
  params_file = os.path.join(al_args.output_dir, f"{al_args.name}_params.yaml")

  args_dict = {}
  for name in sorted(vars(al_args)):
    val = getattr(al_args, name)
    args_dict[name] = val

  with open(params_file, "w") as f:
    yaml.dump(args_dict, f)