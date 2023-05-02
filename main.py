from data import get_test_dataloader
from model import DetectionModel
from train import train
from util import parse_args, set_seed, save_params
from evaluation import generate_predictions, export_to_selection_table, get_metrics, summarize_metrics, predict_and_evaluate

import yaml
import sys
import os

def main(args):
  ## Setup
  args = parse_args(args)
  set_seed(args.seed)
  
  experiment_dir = os.path.join(args.output_dir, args.name)
  setattr(args, 'experiment_dir', str(experiment_dir))
  if not os.path.exists(args.experiment_dir):
    os.makedirs(args.experiment_dir)
  
  save_params(args)
  model = DetectionModel(args)
  
  ## Training
  trained_model = train(model, args)  
  
  ## Evaluation
  test_dataloader = get_test_dataloader(args)
  predict_and_evaluate(trained_model, test_dataloader, args, output_dir = os.path.join(args.experiment_dir, 'test_results'))
 

if __name__ == "__main__":
  main(sys.argv[1:])
  
# python main.py --name=debug --lr=0.0001 --n-epochs=6 --clip-duration=4 --batch-size=100 --omit-empty-clip-prob=0.5 --clip-hop=2
