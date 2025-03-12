import pandas as pd
import yaml
import argparse
import numpy as np
from voxaboxen.evaluation.evaluation import predict_and_generate_manifest, evaluate_based_on_manifest, mean_average_precision
import os

parser = argparse.ArgumentParser()

# General
parser.add_argument('--dset', '-d', type = str, required=True)
parser.add_argument('--seed', type=int, default=0)
ARGS = parser.parse_args()
with open(f'projects/{ARGS.dset}_experiment/project_config.yaml') as f:
    cfg = yaml.load(f, yaml.SafeLoader)

manifest = pd.read_csv(f'../WhisperSeg/outputs/{ARGS.dset}/manifest.txt', sep='\t', index_col=0)
manifest.annotations_fp = [x.removeprefix('../vb3/') for x in manifest.annotations_fp]

manifest.fwd_predictions_fp = [f'../WhisperSeg/{x}' for x in manifest.fwd_predictions_fp]
summary_results = {}
for iou in [0.5, 0.8]:
    test_metrics, test_conf_mats = evaluate_based_on_manifest(manifest, output_dir=None, iou=iou, det_thresh=0.5, class_threshold=0.0, comb_discard_threshold=None, comb_iou_thresh=None, label_mapping=cfg['label_mapping'], unknown_label=cfg['unknown_label'], bidirectional=False)
    summary_results[f'prec@{iou}'] = test_metrics['fwd']['macro']['precision']
    summary_results[f'rec@{iou}'] = test_metrics['fwd']['macro']['recall']
    summary_results[f'f1@{iou}'] = test_metrics['fwd']['macro']['f1']
print(' '.join(f'{k}: {v:.5f}' for k,v in summary_results.items()))

