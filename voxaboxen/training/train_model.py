import numpy as np
import pandas as pd
import torch
from voxaboxen.data.data import get_test_dataloader, get_val_dataloader
from voxaboxen.model.model import DetectionModel
from voxaboxen.training.train import train
from voxaboxen.training.params import parse_args, set_seed, save_params
from voxaboxen.evaluation.evaluation import predict_and_generate_manifest, evaluate_based_on_manifest, mean_average_precision
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

    best_pred_type = 'comb' if args.bidirectional else 'fwd'
    val_manifests_by_thresh = predict_and_generate_manifest(model, get_val_dataloader(args), args, detection_thresholds=np.linspace(0.05, 0.95, 5 if args.is_test else 90), verbose=False)
    best_f1 = 0
    best_thresh = -1
    for det_thresh, manifest in val_manifests_by_thresh.items():
        metrics, _ = evaluate_based_on_manifest(manifest, output_dir=args.experiment_output_dir, results_dir=os.path.join(args.experiment_dir, 'test_results') , iou=0.5, det_thresh=det_thresh, class_threshold=0.0, comb_discard_threshold=args.comb_discard_thresh, label_mapping=args.label_mapping, unknown_label=args.unknown_label, bidirectional=args.bidirectional)
        new_f1 = metrics[best_pred_type]['micro']['f1']
        if new_f1 > best_f1:
            best_f1 = new_f1
            best_thresh = det_thresh

    print(f'Best thresh on val set: {best_thresh:.3f}')

    ## Evaluation
    for split in ['val', 'test']:
        print(f'Evaluating on {split} set')
        if split == 'test':
            test_dataloader = get_test_dataloader(args)
        else:
            test_dataloader = get_val_dataloader(args)
        manifests_by_thresh = predict_and_generate_manifest(model, test_dataloader, args, detection_thresholds=[best_thresh], verbose=False)
        test_manifest = manifests_by_thresh[best_thresh]
        summary_results = {}
        full_results = {}
        for iou in [0.2, 0.5, 0.8]:
            test_metrics, test_conf_mats = evaluate_based_on_manifest(test_manifest, output_dir=args.experiment_output_dir, results_dir=os.path.join(args.experiment_dir, 'test_results') , iou=iou, det_thresh=best_thresh, class_threshold=0.0, comb_discard_threshold=args.comb_discard_thresh, label_mapping=args.label_mapping, unknown_label=args.unknown_label, bidirectional=args.bidirectional)
            full_results[f'f1@{iou}'] = test_metrics
            summary_results[f'micro-f1@{iou}'] = test_metrics[best_pred_type]['micro']['f1']
            summary_results[f'macro-f1@{iou}'] = test_metrics[best_pred_type]['macro']['f1']

        det_thresh_range = np.linspace(0.01, 0.99, 15)
        manifests_by_thresh = predict_and_generate_manifest(model, test_dataloader, args, det_thresh_range, verbose=False)

        for iou in [0.5,0.8]:
            summary_results[f'mean_ap@{iou}'], full_results[f'mAP@{iou}'], full_results[f'ap_by_class@{iou}'] =  mean_average_precision(manifests_by_thresh=manifests_by_thresh, label_mapping=args.label_mapping, exp_dir=args.experiment_dir, iou=iou, pred_type=best_pred_type, comb_discard_thresh=0, bidirectional=args.bidirectional)

        with open(os.path.join(args.experiment_dir, f'{split}_full_results.json'), 'w') as f:
            json.dump(full_results, f)

        with open(os.path.join(args.experiment_dir, f'{split}_results.yaml'), 'w') as f:
            yaml.dump(summary_results, f)

        print(' '.join(f'{k}: {v:.5f}' for k,v in summary_results.items()))
    torch.save(model.state_dict(), os.path.join(args.experiment_dir, 'final-model.pt'))

if __name__ == "__main__":
  train_model(sys.argv[1:])

# python main.py --name=debug --lr=0.0001 --n-epochs=6 --clip-duration=4 --batch-size=100 --omit-empty-clip-prob=0.5 --clip-hop=2
