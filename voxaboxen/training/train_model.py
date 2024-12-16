import numpy as np
import pandas as pd
import torch
from voxaboxen.data.data import get_test_dataloader
from voxaboxen.model.model import DetectionModel
from voxaboxen.training.train import train
from voxaboxen.training.params import parse_args, set_seed, save_params
from voxaboxen.evaluation.evaluation import predict_and_generate_manifest, evaluate_based_on_manifest
import sys
import os
import json
import yaml
import io
from tqdm import tqdm

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

  ## Test F1 @ x
  test_dataloader = get_test_dataloader(args)
  manifests_by_thresh = predict_and_generate_manifest(model, test_dataloader, args)
  test_manifest = manifests_by_thresh[args.detection_threshold]
  best_pred_type = 'comb' if args.bidirectional else 'fwd'
  summary_results = {}
  full_results = {}
  for iou in [0.2, 0.5, 0.8]:
    test_metrics, test_conf_mats = evaluate_based_on_manifest(test_manifest, args, output_dir = os.path.join(args.experiment_dir, 'test_results') , iou=iou, class_threshold=0.0, comb_discard_threshold=args.comb_discard_thresh)
    full_results[f'f1@{iou}'] = test_metrics
    summary_results[f'micro-f1@{iou}'] = test_metrics[best_pred_type]['micro']['f1']
    summary_results[f'macro-f1@{iou}'] = test_metrics[best_pred_type]['macro']['f1']

  ## Test mAP
  det_thresh_range = np.linspace(0.01, 0.99, 15)

  scores_by_class = {c:[] for c in args.label_set}
  manifests_by_thresh = predict_and_generate_manifest(model, test_dataloader, args, det_thresh_range, verbose=False)
  # first loop through thresholds to gather all results
  for det_thresh, test_manifest in manifests_by_thresh.items():
    out_dir = os.path.join(args.experiment_dir, 'mAP', f'detthresh{det_thresh}')
    test_metrics, test_conf_mats = evaluate_based_on_manifest(test_manifest, args, output_dir=out_dir, iou=0.5, class_threshold=0.0, comb_discard_threshold=args.comb_discard_thresh)
    for c, s in test_metrics[best_pred_type]['summary'].items():
        scores_by_class[c].append(dict(s, det_thresh=det_thresh))

  # now loop through classes to calculate APs
  ap_by_class = {}
  full_results['mAP'] = {c:{} for c in args.label_set}
  for c, sweep_ in scores_by_class.items():
    sweep = pd.DataFrame(sweep_).sort_values('recall')
    precs, recs = list(sweep['precision']), list(sweep['recall'])
    precs = [1.] + precs + [0.]
    recs = [0.] + recs + [1.]
    sampled_precs = np.interp(np.linspace(0,1,100), recs, precs)
    ap_by_class[c] = sampled_precs.mean()
    full_results['mAP'][c]['sweep'] = sweep_
    full_results['mAP'][c]['precs'] = precs
    full_results['mAP'][c]['recs'] = recs

  summary_results['mean_ap'] = float(np.array(list(ap_by_class.values())).mean())

  with open(os.path.join(args.experiment_dir, 'full_results.json'), 'w') as f:
    json.dump(full_results, f)

  with open(os.path.join(args.experiment_dir, 'results.yaml'), 'w') as f:
    yaml.dump(summary_results, f)

  print(summary_results)
  torch.save(model.state_dict(), os.path.join(args.experiment_dir, 'final-model.pt'))

if __name__ == "__main__":
  train_model(sys.argv[1:])

# python main.py --name=debug --lr=0.0001 --n-epochs=6 --clip-duration=4 --batch-size=100 --omit-empty-clip-prob=0.5 --clip-hop=2
