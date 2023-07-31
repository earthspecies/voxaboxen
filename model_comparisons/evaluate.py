import sys
sys.path.append("/home/jupyter/sound_event_detection/")

import os
import sys
import argparse
from glob import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader

from detectron2.config import CfgNode
from detectron2.modeling import build_model

from dataloaders import DetectronSingleClipDataset

def evaluate(args):

    cfg = CfgNode(init_dict=CfgNode.load_yaml_with_base(args.full_param_fp))

    list_of_weights = glob(cfg.OUTPUT_DIR + "/*.pth")
    cfg.MODEL.WEIGHTS = max(list_of_weights, key=os.path.getctime) 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold_test if args.score_threshold_test is not None else cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_thresh_test if args.nms_thresh_test is not None else cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST

    model = build_model(cfg)
    model.eval()

    files_to_infer = pd.read_csv(args.file_info_for_inference)
    for row_idx, row in files_to_infer.iterrows():
        audio_fp = row['audio_fp']
        fn = row['fn']

        dataset = DetectronSingleClipDataset(cfg, audio_fp, cfg.SOUND_EVENT.clip_duration/2, cfg.SOUND_EVENT)
        dataloader = DataLoader(
            dataset,
            batch_size = cfg.SOUND_EVENT.batch_size,
            shuffle=False,
            num_workers=cfg.SOUND_EVENT.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        all_boxes = []
        all_scores = []
        all_classes = []
        all_time_bins = []

        with torch.no_grad():
            for d in dataloader:
                ## Run prediction
                #TODO: Batch evaluation? Right now only takes a single record
                assert isinstance(d, dict)
                outputs = model([d])[0]
                instances = outputs["instances"]
                n_mel_freq_bins, n_time_bins = instances.image_size
                boxes, scores, classes = (instances.pred_boxes, instances.scores, instances.pred_classes)
                all_boxes.append(boxes); all_scores.append(scores); all_classes.append(classes); all_time_bins.append(n_time_bins)

            boxes_without_edges = []; scores_without_edges = []; classes_without_edges = []
            for clip_idx, (boxes, scores, classes, n_time_bins) in enumerate(zip(all_boxes, all_scores, all_classes, all_time_bins)):
                ## We use half-overlapping windows, must merge across crops of the sound
                # Compare https://github.com/earthspecies/sound_event_detection/blob/40817cff603d035ba328c49a8b654b7e6171eb3d/source/evaluation/evaluation.py#L141
                # https://github.com/ultralytics/yolov5/issues/11821

                xmin, xmax = boxes.tensor[:, 0].cpu(), boxes.tensor[:, 2].cpu()
                ymin, ymax = boxes.tensor[:, 1].cpu(), boxes.tensor[:, 3].cpu()
                tmin = (xmin/n_time_bins)*float(dataloader.dataset.clip_duration)
                tmax = (xmax/n_time_bins)*float(dataloader.dataset.clip_duration)
                # TODO: could be done more carefully, but we aren't using these for evaluation:
                fmin = dataloader.dataset.spectrogram_f[ymin.round().to(int)]
                fmax = dataloader.dataset.spectrogram_f[ymax.round().to(int)]
                



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-info-for-inference', type=str, required=True, help = "filepath of info csv listing filenames and filepaths of audio for inference")
    parser.add_argument('--full-param-fp', type=str, required=True, help = "filepath of model params saved as a yaml")
    parser.add_argument('--score-threshold-test', type=float, required=False, help = "Confidence score required to get onto the selection table. Defaults to config file.")
    parser.add_argument('--nms-thresh-test', type=float, required=False, help="NMS score threshold to get onto the selection table. Defaults to config file.")
    args = parser.parse_args(sys.argv[1:])  
    evaluate(args)