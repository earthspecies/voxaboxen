import sys
sys.path.append("/home/jupyter/sound_event_detection/")

import os
import sys
import argparse
from glob import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from detectron2.config import CfgNode
from detectron2.modeling import build_model
from torchvision.ops import boxes as box_ops

from dataloaders import DetectronSingleClipDataset
from source.evaluation.evaluation import bbox2raven, write_tsv, evaluate_based_on_manifest
from nms import nms

def evaluate(args):

    cfg = CfgNode(init_dict=CfgNode.load_yaml_with_base(args.full_param_fp))

    list_of_weights = glob(cfg.OUTPUT_DIR + "/*.pth")
    cfg.MODEL.WEIGHTS = max(list_of_weights, key=os.path.getctime) 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold_test if args.score_threshold_test is not None else cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_thresh_test if args.nms_thresh_test is not None else cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    inference_dir = os.path.join(cfg.SOUND_EVENT.experiment_dir, 'inferences')
    if not os.path.isdir(inference_dir):
        os.makedirs(inference_dir)

    model = build_model(cfg)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    files_to_infer = pd.read_csv(args.file_info_for_inference)
    target_fps = []
    for row_idx, row in files_to_infer.iterrows():
        audio_fp = row['audio_fp']
        fn = row['fn']

        dataset = DetectronSingleClipDataset(cfg, audio_fp, cfg.SOUND_EVENT.clip_duration/2, cfg.SOUND_EVENT)
        dataloader = DataLoader(
            dataset,
            batch_size = 1, #TODO: cfg.SOUND_EVENT.batch_size,
            shuffle=False,
            num_workers=cfg.SOUND_EVENT.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        boxes, scores, classes, len_t = generate_predictions(model, dataloader)
        #TODO account for whether to keep arrays as numpy or torch
        events, boxes, scores, classes = account_for_overlapping_clips(boxes, scores, classes, len_t, dataloader)
        if args.time_based_nms:
            #boxes shape: (n_boxes, 2=(onset,offset))
            events, keep_indices = nms(events, scores, iou_thresh=cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
            scores = scores[keep_indices]; classes = classes[keep_indices]
        else:
            if args.ignore_interclass_iou: 
                #boxes shape: (n_boxes, 4=(xmin,ymin,xmax,ymax))
                keep_indices = box_ops.batched_nms(torch.Tensor(boxes), torch.Tensor(scores), torch.Tensor(classes), iou_threshold=cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
            else:
                keep_indices = box_ops.nms(torch.Tensor(boxes), torch.Tensor(scores), iou_threshold=cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)   
            keep_indices = sorted(keep_indices.numpy()) #Will be sorted by score, now re-sort by time
            events = events[keep_indices, :]; scores = scores[keep_indices]; classes = classes[keep_indices]   

        #TODO Currently ignores class probabilities, see https://github.com/facebookresearch/detectron2/issues/1754
        target_fp = os.path.join(inference_dir, f"peaks_pred_{fn}.txt")
        st = bbox2raven(events, classes, cfg.SOUND_EVENT.label_set, scores, np.full(classes.shape, 1), "Unknown")
        write_tsv(target_fp, st)
        print(f"Saving predictions for {fn} to {target_fp}")
        target_fps.append(target_fp)
    
    ## If we have a dataset with existing manual annotations, compute metrics against these annotations
    if ("selection_table_fp" in files_to_infer.columns) or ("annotations_fp" in files_to_infer.columns):
        print("Calculating metrics against manual annotations.")
        #TODO why are these different than how it was originally set up?
        files_to_infer["filename"] = files_to_infer["fn"]
        files_to_infer["predictions_fp"] = target_fps
        if "selection_table_fp" in files_to_infer.columns:
            files_to_infer["annotations_fp"] = files_to_infer["selection_table_fp"]
        for iou in [0.2, 0.5, 0.8]:
            for class_threshold in [0.0, 0.5, 0.95]:
                evaluate_based_on_manifest(
                    files_to_infer, 
                    cfg.SOUND_EVENT, 
                    output_dir = os.path.join(cfg.SOUND_EVENT.experiment_dir, 'test_results'), 
                    iou = iou, 
                    class_threshold = class_threshold
                    ) 
    
def generate_predictions(model, dataloader):
    boxes = []; scores = []; classes = []; len_t = []
    with torch.no_grad():
        ## Run prediction for all the clips corresponding to one file
        for d in dataloader:
            #Prepare input dict
            assert isinstance(d, dict)
            if torch.cuda.is_available():
                d["image"] = d["image"].cuda()
            d["image"] = d["image"].squeeze(0) #model does not take batch dimension
            d["height"] = d["height"].item(); d["width"] = d["width"].item()
            #TODO: Batch evaluation? Right now only takes a single record
            outputs = model([d])[0]
            instances = outputs["instances"]
            n_mel_freq_bins, n_time_frames = instances.image_size
            boxes.append(instances.pred_boxes); scores.append(instances.scores); 
            classes.append(instances.pred_classes); len_t.append(n_time_frames)
    return boxes, scores, classes, len_t

def account_for_overlapping_clips(boxes, scores, classes, len_t, dataloader):
    """ Account for overlap in clips
    We use half-overlapping windows, must merge across crops of the sound 
    TODO Could potentially be more careful, e.g. https://github.com/ultralytics/yolov5/issues/11821
    """
    events_without_edges = []; boxes_without_edges = []; scores_without_edges = []; classes_without_edges = []
    for clip_idx, (boxes, scores, classes, n_time_bins) in enumerate(zip(boxes, scores, classes, len_t)):
        # Compare https://github.com/earthspecies/sound_event_detection/blob/40817cff603d035ba328c49a8b654b7e6171eb3d/source/evaluation/evaluation.py#L141
        xmin = boxes.tensor[:, 0].cpu()
        if clip_idx == 0 and len(boxes) == 1:
            include_boxes = xmin >= 0
        elif clip_idx == 0 and len(boxes) > 1:
            first_quarter = n_time_bins//4
            last_quarter = n_time_bins - (n_time_bins//2 - first_quarter)
            include_boxes = xmin < last_quarter                
        elif clip_idx == len(boxes) - 1:
            include_boxes = xmin >= first_quarter
        else:
            include_boxes = (xmin < last_quarter) * (xmin >= first_quarter)
        xmin, xmax = boxes.tensor[include_boxes, 0].cpu(), boxes.tensor[include_boxes, 2].cpu()
        ymin, ymax = boxes.tensor[include_boxes, 1].cpu(), boxes.tensor[include_boxes, 3].cpu()
        # TODO: frequency determination could be done more carefully, but we aren't using these for evaluation
        tmin = ((xmin/n_time_bins)*float(dataloader.dataset.clip_duration) + clip_idx*dataloader.dataset.clip_hop).numpy()
        tmax = ((xmax/n_time_bins)*float(dataloader.dataset.clip_duration) + clip_idx*dataloader.dataset.clip_hop).numpy()
        # fmin = dataloader.dataset.spectrogram_f[ymin.round().to(int).clamp(min=0,max=len(dataloader.dataset.spectrogram_f)-1)]
        # fmax = dataloader.dataset.spectrogram_f[ymax.round().to(int).clamp(min=0,max=len(dataloader.dataset.spectrogram_f)-1)]
        # Also get scores and classes
        events_without_edges.append(np.stack([tmin, tmax])) #Different from Detectron format which is (tmin, fmin, tmax, fmax)
        boxes_without_edges.append(np.stack([xmin, ymin, xmax, ymax]))
        scores_without_edges.append(scores[include_boxes].cpu().numpy()); 
        classes_without_edges.append(classes[include_boxes].cpu().numpy())

    events = np.concatenate(events_without_edges, axis=1).T
    boxes = np.concatenate(boxes_without_edges, axis=1).T
    scores = np.concatenate(scores_without_edges)
    classes = np.concatenate(classes_without_edges)

    return events, boxes, scores, classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-info-for-inference', type=str, required=True, help = "filepath of info csv listing filenames and filepaths of audio for inference")
    parser.add_argument('--full-param-fp', type=str, required=True, help = "filepath of model params saved as a yaml")
    parser.add_argument('--score-threshold-test', type=float, required=False, help = "Confidence score required to get onto the selection table. Defaults to config file.")
    parser.add_argument('--nms-thresh-test', type=float, required=False, help="NMS score threshold to get onto the selection table. Defaults to config file.")
    parser.add_argument('--time-based-nms', action="store_true", help = "If true, compute NMS only based on onset/offset but ignore frequency.")
    parser.add_argument('--ignore-interclass-iou', action="store_true", help="If true, ignore overlap between boxes of different classes.")
    args = parser.parse_args(sys.argv[1:])  
    evaluate(args)