import pandas as pd
from voxaboxen.data.data import get_test_dataloader
import torch
from voxaboxen.model.model import DetectionModel, DetectionModelStereo
from voxaboxen.training.train import train
from voxaboxen.training.params import parse_args, set_seed, save_params
from voxaboxen.evaluation.evaluation import generate_predictions, export_to_selection_table, get_metrics, summarize_metrics, predict_and_generate_manifest, evaluate_based_on_manifest

import yaml
import sys
import os

def train_model(args):
  ## Setup
  args = parse_args(args)

  set_seed(args.seed)

  experiment_dir = os.path.join(args.project_dir, args.name)
  setattr(args, 'experiment_dir', str(experiment_dir))
  if os.path.exists(args.experiment_dir) and not args.overwrite:
    sys.exit('experiment already exists with this name')
    os.makedirs(args.experiment_dir)

  experiment_output_dir = os.path.join(experiment_dir, "outputs")
  setattr(args, 'experiment_output_dir', experiment_output_dir)
  if not os.path.exists(args.experiment_output_dir):
    os.makedirs(args.experiment_output_dir)

  save_params(args)
  if hasattr(args,'stereo') and args.stereo:
    model = DetectionModelStereo(args)
  else:
    model = DetectionModel(args)

  if args.reload_from is not None:
    #model.load_state_dict(os.path.join(args.experiment_dir), 'model.pt')
    checkpoint = torch.load(os.path.join(args.project_dir, args.reload_from, 'model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

  ## Training
  if args.n_epochs == 0:
    trained_model = model
  else:
      trained_model = train(model, args)

  ## Evaluation
  test_dataloader = get_test_dataloader(args)

  manifest = predict_and_generate_manifest(trained_model, test_dataloader, args)

  #class_threshes = [0] if len(args.label_set)==1 else [0.0, 0.5, 0.95]
  class_threshes = [0.0, 0.5, 0.95]
  for iou in [0.2, 0.5, 0.8]:
    for class_threshold in class_threshes:
        for comb_discard_thresh in [0.85]:
          metrics, conf_mats = evaluate_based_on_manifest(manifest, args, output_dir = os.path.join(args.experiment_dir, 'test_results') , iou=iou, class_threshold=class_threshold, comb_discard_threshold=comb_discard_thresh)
          print(f'IOU: {iou} class_thresh: {class_threshold} Comb discard threshold: {comb_discard_thresh}')
          for pred_type in metrics.keys():
              to_print = {k1:{k:round(100*v,4) for k,v in v1.items()} for k1,v1 in metrics[pred_type]['summary'].items()} if len(args.label_set)==1 else dict(pd.DataFrame(metrics[pred_type]['summary']).mean(axis=1).round(4))
              print(f'{pred_type}:', to_print)

if __name__ == "__main__":
  train_model(sys.argv[1:])

# python main.py --name=debug --lr=0.0001 --n-epochs=6 --clip-duration=4 --batch-size=100 --omit-empty-clip-prob=0.5 --clip-hop=2
