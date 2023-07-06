from source.data.data import get_test_dataloader
from source.model.model import DetectionModel
from source.training.train import train
from source.training.params import parse_args, set_seed, save_params
from source.evaluation.evaluation import generate_predictions, export_to_selection_table, get_metrics, summarize_metrics, predict_and_generate_manifest, evaluate_based_on_manifest

import yaml
import sys
import os

def train_model(args):
  ## Setup
  args = parse_args(args)

  set_seed(args.seed)
  
  experiment_dir = os.path.join(args.project_dir, args.name)
  setattr(args, 'experiment_dir', str(experiment_dir))
  if not os.path.exists(args.experiment_dir):
    os.makedirs(args.experiment_dir)
    
  experiment_output_dir = os.path.join(experiment_dir, "outputs")
  setattr(args, 'experiment_output_dir', experiment_output_dir)
  if not os.path.exists(args.experiment_output_dir):
    os.makedirs(args.experiment_output_dir)
  
  save_params(args)
  model = DetectionModel(args)
  
  ## Training
  import torch
  if args.previous_checkpoint_fp is not None:
    print(f"loading model weights from {args.previous_checkpoint_fp}")
    cp = torch.load(args.previous_checkpoint_fp)
    model.load_state_dict(cp["model_state_dict"])
  trained_model = model
  # trained_model = train(model, args)  
  
  ## Evaluation
  test_dataloader = get_test_dataloader(args)
  
  manifest = predict_and_generate_manifest(trained_model, test_dataloader, args)
  
  for iou in [0.2, 0.5, 0.8]:
    for class_threshold in [0.0, 0.5, 0.95]:
      evaluate_based_on_manifest(manifest, args, output_dir = os.path.join(args.experiment_dir, 'test_results') , iou = iou, class_threshold = class_threshold) 

if __name__ == "__main__":
  train_model(sys.argv[1:])
  
# python main.py --name=debug --lr=0.0001 --n-epochs=6 --clip-duration=4 --batch-size=100 --omit-empty-clip-prob=0.5 --clip-hop=2
