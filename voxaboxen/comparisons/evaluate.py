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
from tqdm import tqdm

from detectron2.config import CfgNode
from detectron2.modeling import build_model
from torchvision.ops import boxes as box_ops

from voxaboxen.comparisons.dataloaders import DetectronSingleClipDataset, SoundEventTrainer
from voxaboxen.evaluation.evaluation import bbox2raven, write_tsv, evaluate_based_on_manifest
from voxaboxen.comparisons.nms import nms

def if_not_none(x, y):
    return x if x is not None else y

def evaluate_from_cfg(args):
    """ Process a detectron/sound event config to run evaluation on its specified model """
    cfg = CfgNode(init_dict=CfgNode.load_yaml_with_base(args.full_param_fp))
    list_of_weights = glob(cfg.OUTPUT_DIR + "/*.pth")
    cfg.MODEL.WEIGHTS = max(list_of_weights, key=os.path.getctime) 
    print("Loading ", cfg.MODEL.WEIGHTS)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = if_not_none(args.score_threshold_test, cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = if_not_none(args.nms_thresh_test, cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
    cfg.SOUND_EVENT.EVAL.IGNORE_INTERCLASS_IOU = if_not_none(args.ignore_interclass_iou, cfg.SOUND_EVENT.EVAL.IGNORE_INTERCLASS_IOU)
    cfg.SOUND_EVENT.EVAL.TIME_BASED_NMS = if_not_none(args.time_based_nms, cfg.SOUND_EVENT.EVAL.TIME_BASED_NMS)
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)["model"])
    run_evaluation(model, args.file_info_for_inference, cfg, args.results_folder_name)
    
def run_evaluation(model, inference_fp, cfg, results_folder_name):
    """ Run model evaluation on files indicated in inference_fp, save metrics in path specified with results_folder_name """

    inference_dir = os.path.join(cfg.SOUND_EVENT.experiment_dir, 'inferences')
    if not os.path.isdir(inference_dir):
        os.makedirs(inference_dir)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    target_fps = []
    durations = []
    files_to_infer = pd.read_csv(inference_fp)
    if len(files_to_infer) == 0:
        print(f"No sound files specified in {inference_fp}")
        return 
    for row_idx, row in tqdm(files_to_infer.iterrows(),total=len(files_to_infer)):
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

        # Calculate duration based on clip parameters
        num_clips = len(dataloader.dataset)
        clip_duration = dataset.clip_duration
        clip_hop = dataset.clip_hop
        duration_sec = (num_clips - 1) * clip_hop + clip_duration
        durations.append(duration_sec)

        boxes, scores, classes, len_t = generate_predictions(model, dataloader)
        #TODO (low priority) clean up by accounting for whether to keep arrays as numpy or torch?
        events, boxes, scores, classes = remove_edge_boxes(boxes, scores, classes, len_t, dataloader)
        if events is not None:
            if cfg.SOUND_EVENT.EVAL.TIME_BASED_NMS:
                #boxes shape: (n_boxes, 2=(onset,offset))
                events, keep_indices = nms(events, scores, iou_thresh=cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
                scores = scores[keep_indices]; classes = classes[keep_indices]
            else:
                if cfg.SOUND_EVENT.EVAL.IGNORE_INTERCLASS_IOU: 
                    #boxes shape: (n_boxes, 4=(xmin,ymin,xmax,ymax))
                    keep_indices = box_ops.batched_nms(torch.Tensor(boxes), torch.Tensor(scores), torch.Tensor(classes), iou_threshold=cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
                else:
                    keep_indices = box_ops.nms(torch.Tensor(boxes), torch.Tensor(scores), iou_threshold=cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)   
                keep_indices = sorted(keep_indices.numpy()) #Will be sorted by score, now re-sort by time
                events = events[keep_indices, :]; scores = scores[keep_indices]; classes = classes[keep_indices]   

        #TODO Currently does not attempt to retrieve class probabilities, see https://github.com/facebookresearch/detectron2/issues/1754
        target_fp = os.path.join(inference_dir, f"peaks_pred_{fn}.txt")
        st = bbox2raven(events, classes, cfg.SOUND_EVENT.label_set, scores, np.full(classes.shape, 1) if classes is not None else None, "Unknown")
        write_tsv(target_fp, st)
        print(f"Saving predictions for {fn} to {target_fp}")
        target_fps.append(target_fp)

    files_to_infer["duration_sec"] = durations

    ## If we have a dataset with existing manual annotations, compute metrics against these annotations
    if ("selection_table_fp" in files_to_infer.columns) or ("annotations_fp" in files_to_infer.columns):
        print("Calculating metrics against manual annotations.")
        #TODO why are these different than how it was originally set up?
        files_to_infer["filename"] = files_to_infer["fn"]
        files_to_infer["fwd_predictions_fp"] = target_fps
        if "selection_table_fp" in files_to_infer.columns:
            files_to_infer["annotations_fp"] = files_to_infer["selection_table_fp"]
        for iou in [0.2, 0.5, 0.8]:
            # Currently do not have class probabilities < 1 for detectron output (see above), so these are redundant
            for class_threshold in [0.0, 0.5, 0.95]:  
                evaluate_based_on_manifest(
                    files_to_infer, 
                    cfg.SOUND_EVENT, 
                    output_dir = os.path.join(cfg.SOUND_EVENT.experiment_dir, results_folder_name), 
                    iou = iou, 
                    class_threshold = class_threshold
                    ) 
    
def generate_predictions(model, dataloader):
    """ Run all clips stemming from a single sound file through model """
    boxes = []; scores = []; classes = []; len_t = []
    with torch.no_grad():
        ## Run prediction for all the clips corresponding to one file
        for d in dataloader:
            #Prepare input dict
            assert isinstance(d, dict)
            if torch.cuda.is_available():
                d["image"] = d["image"].cuda() 
            #TODO (low priority): Batch evaluation? Right now only takes a single record, where image has no batch dimension
            outputs = model([{
                "image": d["image"].squeeze(0), 
                "width":d["width"].item(), 
                "height": d["height"].item()
                }])[0]
            instances = outputs["instances"]
            n_mel_freq_bins, n_time_frames = instances.image_size
            boxes.append(instances.pred_boxes); scores.append(instances.scores); 
            classes.append(instances.pred_classes); len_t.append(n_time_frames)
    return boxes, scores, classes, len_t

def remove_edge_boxes(list_boxes, list_scores, list_classes, list_len_t, dataloader):
    """ Get rid of boxes which are on the edges of clips
        We use half-overlapping windows, must merge across crops of the sound 
        TODO Could potentially be more careful, e.g. https://github.com/ultralytics/yolov5/issues/11821
    """
    events_without_edges = []; boxes_without_edges = []; scores_without_edges = []; classes_without_edges = []
    n_time_bins = list_len_t[0]
    first_quarter = n_time_bins//4
    last_quarter = n_time_bins - (n_time_bins//2 - first_quarter)
    for clip_idx in range(len(dataloader.dataset)):
        # Compare https://github.com/earthspecies/sound_event_detection/blob/40817cff603d035ba328c49a8b654b7e6171eb3d/source/evaluation/evaluation.py#L141
        boxes = list_boxes[clip_idx]
        scores = list_scores[clip_idx]
        classes = list_classes[clip_idx]
        n_time_bins = list_len_t[clip_idx]
        if len(boxes) == 0:
            continue
        xmin = boxes.tensor[:, 0].cpu()
        if clip_idx == 0:
            if len(dataloader.dataset) > 1:
                include_boxes = xmin < last_quarter
            else:
                # Include all boxes in first clip, if there's only one clip for this sound.
                include_boxes = xmin > 0           
        elif clip_idx == len(dataloader.dataset) - 1:
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

    if len(events_without_edges) == 0:
        return None, None, None, None
    else:
        events = np.concatenate(events_without_edges, axis=1).T
        boxes = np.concatenate(boxes_without_edges, axis=1).T
        scores = np.concatenate(scores_without_edges)
        classes = np.concatenate(classes_without_edges)
        return events, boxes, scores, classes

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-info-for-inference', type=str, required=True, help = "filepath of info csv listing filenames and filepaths of audio for inference")
    parser.add_argument('--full-param-fp', type=str, required=True, help = "filepath of model params saved as a yaml")
    parser.add_argument('--results-folder-name', type=str, required=False, default="test_results", help = "Name of subfolder in experiment dir.")
    parser.add_argument('--score-threshold-test', type=float, required=False, help = "Confidence score required to get onto the selection table. Defaults to config file.")
    parser.add_argument('--nms-thresh-test', type=float, required=False, help="NMS score threshold to get onto the selection table. Defaults to config file.")
    parser.add_argument('--time-based-nms', action="store_true", help = "If true, compute NMS only based on onset/offset but ignore frequency.")
    parser.add_argument('--ignore-interclass-iou', action="store_true", help="If true, ignore overlap between boxes of different classes.")
    args = parser.parse_args(args)
    return args

if __name__ == "__main__":
    evaluate_from_cfg(parse_args(sys.argv[1:]))