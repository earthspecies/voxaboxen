from time import time
import shutil
import numpy as np
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
from fn_profiling import profile_lines

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(args):
    ## Setup
    args = parse_args(args)

    set_seed(args.seed)

    experiment_dir = os.path.join(args.project_dir, args.name)

    if os.path.exists(os.path.join(experiment_dir, 'model.pt')):
        if args.exists_strategy=='resume':
            args.previous_checkpoint_fp = os.path.join(experiment_dir, 'model.pt')
            with open(os.path.join(experiment_dir, 'train_history.yaml')) as f:
                x = yaml.load(f, Loader=yaml.SafeLoader)
            n_epochs_ran_for = len(x)
            args.n_epochs -= n_epochs_ran_for
            print(f'resuming previous run which ran for {n_epochs_ran_for} epochs, now training for the remaining {args.n_epochs}')
            assert max(x.keys()) == n_epochs_ran_for-1
            args.unfreeze_encoder_epoch = max(0, args.unfreeze_encoder_epoch-n_epochs_ran_for)

        elif args.exists_strategy == 'none' and args.name!='demo':
            sys.exit('experiment already exists with this name')

    experiment_output_dir = os.path.join(experiment_dir, "outputs")
    if not os.path.exists(experiment_output_dir):
      os.makedirs(experiment_output_dir)

    setattr(args, 'experiment_dir', str(experiment_dir))
    setattr(args, 'experiment_output_dir', experiment_output_dir)
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

    val_fit_starttime = time()
    best_pred_type = 'comb' if args.bidirectional else 'fwd'
    val_manifest = predict_and_generate_manifest(model, get_val_dataloader(args), args, verbose=False)[0.5]
    best_f1 = 0
    best_comb_discard = -1
    os.makedirs(match_cache_dir:=os.path.join(experiment_output_dir, 'tmp-cache'), exist_ok=True)
    for comb_discard in np.linspace(0.0, 0.95, args.n_val_fit):
        #for comb_iou in np.linspace(0.4, 0.8, 8):
        for comb_iou in [0.5]:
            #metrics5, _ = evaluate_based_on_manifest(val_manifest, output_dir=args.experiment_output_dir, iou=0.5, det_thresh=0.5, class_threshold=0.0, comb_discard_threshold=comb_discard, label_mapping=args.label_mapping, unknown_label=args.unknown_label, bidirectional=args.bidirectional, pred_types=(best_pred_type,))
            metrics8, _ = evaluate_based_on_manifest(val_manifest, output_dir=args.experiment_output_dir, iou=0.8, det_thresh=0.5, class_threshold=0.0, comb_discard_threshold=comb_discard, comb_iou_thresh=comb_iou, label_mapping=args.label_mapping, unknown_label=args.unknown_label, bidirectional=args.bidirectional, pred_types=(best_pred_type,))
            #new_f15 = metrics5[best_pred_type]['macro']['f1']
            #new_f18 = metrics8[best_pred_type]['macro']['f1']
            #new_f1 = 2*new_f15*new_f18 / (new_f15+new_f18)
            new_f1 = metrics8[best_pred_type]['macro']['f1']
            if new_f1 > best_f1:
                best_f1 = new_f1
                best_comb_discard = comb_discard
                best_comb_iou = comb_iou

    print(f'Found best thresh on val set: f1={best_f1:.4f}, comb_discard={best_comb_discard:.3f}, comb_iou={best_comb_iou:.3f} in {time()-val_fit_starttime:.3f}s')

    ## Evaluation
    val_fit_starttime = time()
    for split in ['val', 'test']:
        print(f'Evaluating on {split} set')
        if split == 'test':
            test_dataloader = get_test_dataloader(args)
        else:
            test_dataloader = get_val_dataloader(args)
        manifests_by_thresh = predict_and_generate_manifest(model, test_dataloader, args, verbose=False)
        print(f'Time to compute manifests_by_thresh: {time()-val_fit_starttime:.3f}s')
        test_manifest = manifests_by_thresh[0.5]
        summary_results = {}
        full_results = {}
        eval_starttime = time()
        #for iou in [0.2, 0.5, 0.8]:
        for iou in [0.5, 0.8]:
            test_metrics, test_conf_mats = evaluate_based_on_manifest(test_manifest, output_dir=experiment_output_dir, iou=iou, det_thresh=0.5, class_threshold=0.0, comb_discard_threshold=best_comb_discard, comb_iou_thresh=best_comb_iou, label_mapping=args.label_mapping, unknown_label=args.unknown_label, bidirectional=args.bidirectional)
            full_results[f'f1@{iou}'] = test_metrics
            summary_results[f'micro-f1@{iou}'] = test_metrics[best_pred_type]['micro']['f1']
            summary_results[f'macro-f1@{iou}'] = test_metrics[best_pred_type]['macro']['f1']

        print(f'Time to compute f1s: {time()-eval_starttime:.3f}s')
        if args.bidirectional:
            det_thresh_range = np.concatenate([np.linspace(0.001, 0.2, args.n_map//3), np.linspace(0.21, 0.7, args.n_map//3)]) # to make sure the lower range is covered, sweep fewer overall cus sweep other threshes too
        else:
            det_thresh_range = np.linspace(0.01, 0.99, args.n_map)
        manifests_by_thresh = predict_and_generate_manifest(model, test_dataloader, args, det_thresh_range, verbose=False)

        map_starttime = time()
        for iou in [0.5, 0.8]:
            if split=='val' and iou==0.8:
                continue
            summary_results[f'mean_ap@{iou}'], full_results[f'mAP@{iou}'], full_results[f'ap_by_class@{iou}'] =  mean_average_precision(manifests_by_thresh=manifests_by_thresh, label_mapping=args.label_mapping, exp_dir=experiment_dir, iou=iou, pred_type=best_pred_type, bidirectional=args.bidirectional, comb_iou_thresh=best_comb_iou)

        with open(os.path.join(args.experiment_dir, f'{split}_full_results.json'), 'w') as f:
            json.dump(full_results, f)

        with open(os.path.join(args.experiment_dir, f'{split}_results.yaml'), 'w') as f:
            yaml.dump(summary_results, f)

        print(f'time to compute mAP: {time()-map_starttime:.3f}')
        print(' '.join(f'{k}: {v:.5f}' for k,v in summary_results.items()))
    shutil.rmtree(match_cache_dir)
    torch.save(model.state_dict(), os.path.join(args.experiment_dir, 'final-model.pt'))

if __name__ == "__main__":
  train_model(sys.argv[1:])
