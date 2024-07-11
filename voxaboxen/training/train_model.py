import pandas as pd
import torch
from voxaboxen.data.data import get_test_dataloader
from voxaboxen.model.model import DetectionModel
from voxaboxen.training.train import train
from voxaboxen.training.params import parse_args, set_seed, save_params
from voxaboxen.evaluation.evaluation import predict_and_generate_manifest, evaluate_based_on_manifest
import sys
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

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
  if os.path.exists(args.experiment_dir) and (not args.overwrite) and args.name!='demo':
    sys.exit('experiment already exists with this name')

  experiment_output_dir = os.path.join(experiment_dir, "outputs")
  setattr(args, 'experiment_output_dir', experiment_output_dir)
  if not os.path.exists(args.experiment_output_dir):
    os.makedirs(args.experiment_output_dir)

  save_params(args)
  model = DetectionModel(args).to(device)
    
  if args.previous_checkpoint_fp is not None:
    print(f"loading model weights from {args.previous_checkpoint_fp}")
    cp = torch.load(args.previous_checkpoint_fp)
    if "model_state_dict" in cp.keys():
      model.load_state_dict(cp["model_state_dict"])
    else:
      model.load_state_dict(cp)

  ## Training
  if args.n_epochs>0:
    model = train(model, args)
  
  test_dataloader = get_test_dataloader(args)
  test_manifest = predict_and_generate_manifest(model, test_dataloader, args)
  for iou in [0.2, 0.5, 0.8]:
    test_metrics, test_conf_mats = evaluate_based_on_manifest(test_manifest, args, output_dir = os.path.join(args.experiment_dir, 'test_results') , iou=iou, class_threshold=0.5, comb_discard_threshold=args.comb_discard_thresh)
    print(f'Test with IOU{iou}')
    print_metrics(test_metrics, just_one_label=(len(args.label_set)==1))

  torch.save(model.state_dict(), os.path.join(args.experiment_dir, 'final-model.pt'))

if __name__ == "__main__":
  train_model(sys.argv[1:])

# python main.py --name=debug --lr=0.0001 --n-epochs=6 --clip-duration=4 --batch-size=100 --omit-empty-clip-prob=0.5 --clip-hop=2
