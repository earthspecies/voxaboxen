import pandas as pd
from voxaboxen.data.data import get_test_dataloader, get_val_dataloader
import torch
from voxaboxen.model.model import DetectionModel, DetectionModelStereo
from voxaboxen.training.train import train
from voxaboxen.training.params import parse_args, set_seed, save_params
from voxaboxen.evaluation.evaluation import generate_predictions, export_to_selection_table, get_metrics, summarize_metrics, predict_and_generate_manifest, evaluate_based_on_manifest

import yaml
import sys
import os


def print_metrics(metrics, just_one_label):
    for pred_type in metrics.keys():
        to_print = {k1:{k:round(100*v,4) for k,v in v1.items()} for k1,v1 in metrics[pred_type]['summary'].items()} if just_one_label else dict(pd.DataFrame(metrics[pred_type]['summary']).mean(axis=1).round(4))
        print(f'{pred_type}:', to_print)
def train_model(args):

  ## Setup
  args = parse_args(args)

  set_seed(args.seed)

  experiment_dir = os.path.join(args.project_dir, args.name)
  setattr(args, 'experiment_dir', str(experiment_dir))
  if os.path.exists(args.experiment_dir) and args.overwrite and args.name!='demo':
    sys.exit('experiment already exists with this name')
    os.makedirs(args.experiment_dir)

  experiment_output_dir = os.path.join(experiment_dir, "outputs")
  setattr(args, 'experiment_output_dir', experiment_output_dir)
  if not os.path.exists(args.experiment_output_dir):
    os.makedirs(args.experiment_output_dir)

  save_params(args)
  model = DetectionModel(args)

  if args.reload_from is not None:
    checkpoint = torch.load(os.path.join(args.project_dir, args.reload_from, 'model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

  ## Training
  trained_model = train(model, args)

  ## Evaluation
  test_dataloader = get_test_dataloader(args)
  val_dataloader = get_val_dataloader(args)

  val_manifest = predict_and_generate_manifest(trained_model, val_dataloader, args)

  model.comb_discard_thresh = -1
  if model.is_bidirectional:
      best_f1 = 0
      for comb_discard_thresh in [.3,.35,.4,.45,.5,.55,.6,.65,.75,.8,.85,.9]:
        val_metrics, val_conf_mats = evaluate_based_on_manifest(val_manifest, args, output_dir = os.path.join(args.experiment_dir, 'test_results') , iou=0.5, class_threshold=0.5, comb_discard_threshold=comb_discard_thresh)
        new_f1 = val_metrics['comb']['macro']['f1']
        if new_f1 > best_f1:
          model.comb_discard_thresh = comb_discard_thresh
          best_f1 = new_f1
        print(f'IOU: 0.5 class_thresh: 0.5 Comb discard threshold: {comb_discard_thresh}')
        print_metrics(val_metrics, just_one_label=(len(args.label_set)==1))
      print(f'Using comb_discard_thresh: {model.comb_discard_thresh}')

  test_manifest = predict_and_generate_manifest(trained_model, test_dataloader, args)
  for iou in [0.2, 0.5, 0.8]:
    test_metrics, test_conf_mats = evaluate_based_on_manifest(test_manifest, args, output_dir = os.path.join(args.experiment_dir, 'test_results') , iou=iou, class_threshold=0.5, comb_discard_threshold=model.comb_discard_thresh)
    print(f'Test with IOU{iou}')
    print_metrics(test_metrics, just_one_label=(len(args.label_set)==1))

if __name__ == "__main__":
  train_model(sys.argv[1:])

# python main.py --name=debug --lr=0.0001 --n-epochs=6 --clip-duration=4 --batch-size=100 --omit-empty-clip-prob=0.5 --clip-hop=2
