import yaml
from os.path import join
import os
import pandas as pd


all_results = {}
for detthresh in (0.4, 0.55, 0.7, 0.85):
    all_results[detthresh] = {}
    for combiouthresh in (0.4, 0.55, 0.7, 0.85):
        all_results[detthresh][combiouthresh] = {}
        for combdiscardthresh in  (0.4, 0.55, 0.7, 0.85):
            all_results[detthresh][combiouthresh][combdiscardthresh] = {}
            resdir = f'projects/MT_experiment/bidirectional-{detthresh}-{combiouthresh}-{combdiscardthresh}/test_results'
            if not os.path.exists(resdir):
                continue
            results = {}
            for iouf1 in (2,5,8):
                with open(join(resdir, f'metrics_iou_0.{iouf1}_class_threshold_0.yaml')) as f:
                    exp_results = yaml.safe_load(f)
                for pred_type in ('fwd','bck','comb','match'):
                   results[f'testiou{iouf1}-{pred_type}'] = exp_results[pred_type]['macro']['f1']
            all_results[detthresh][combiouthresh][combdiscardthresh] = results

breakpoint()

