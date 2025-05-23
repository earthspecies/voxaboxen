"""
Perform inference using a trained model, and save predictions as selection tables.
"""

import os

import pandas as pd
import torch

from voxaboxen.data.data import get_single_clip_data
from voxaboxen.evaluation.evaluation import (combine_fwd_bck_preds,
                                             export_to_selection_table,
                                             generate_predictions)
from voxaboxen.inference.params import parse_inference_args
from voxaboxen.model.model import DetectionModel  # , DetectionModelStereo
from voxaboxen.training.params import load_params

device = "cuda" if torch.cuda.is_available() else "cpu"


def inference(inference_args):
    """
    Perform inference using a trained model, and save predictions as selection tables.
    Parameters
    ----------
    inference_args : argparse.Namespace
        Configuration arguments; see params.py
    Returns
    ----------
    """
    inference_args = parse_inference_args(inference_args)
    args = load_params(inference_args.model_args_fp)
    files_to_infer = pd.read_csv(inference_args.file_info_for_inference)

    output_dir = os.path.join(args.experiment_dir, "inference")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = DetectionModel(args)

    if inference_args.model_checkpoint_fp is None:
        model_checkpoint_fp = os.path.join(args.experiment_dir, "final-model.pt")
    else:
        model_checkpoint_fp = inference_args.model_checkpoint_fp

    print(f"Loading model weights from {model_checkpoint_fp}")
    cp = torch.load(model_checkpoint_fp)
    if "model_state_dict" in cp.keys():
        model.load_state_dict(cp["model_state_dict"])
    else:
        model.load_state_dict(cp)
    model = model.to(device)

    for i, row in files_to_infer.iterrows():
        audio_fp = row["audio_fp"]
        fn = row["fn"]

        if not os.path.exists(audio_fp):
            print(f"Could not locate file {audio_fp}")
            continue

        dataloader = None
        try:
            dataloader = get_single_clip_data(audio_fp, args.clip_duration / 2, args)
        except dataloader is None:
            print(f"Could not load file {audio_fp}")
            continue

        if len(dataloader) == 0:
            print(f"Skipping {fn} because it is too short")
            continue

        if inference_args.disable_bidirectional and not model.is_bidirectional:
            print(
                "Warning: you have passed the disable-bidirectional arg but model is not is_bidirectional"
            )
        (
            detections,
            regressions,
            classifs,
            rev_detections,
            rev_regressions,
            rev_classifs,
        ) = generate_predictions(model, dataloader, args, verbose=True)

        fwd_target_fp = export_to_selection_table(
            detections,
            regressions,
            classifs,
            fn,
            args,
            is_bck=False,
            verbose=True,
            target_dir=output_dir,
            detection_threshold=inference_args.detection_threshold,
            classification_threshold=inference_args.classification_threshold,
        )

        if model.is_bidirectional and not inference_args.disable_bidirectional:
            export_to_selection_table(
                rev_detections,
                rev_regressions,
                rev_classifs,
                fn,
                args,
                is_bck=True,
                verbose=True,
                target_dir=output_dir,
                detection_threshold=inference_args.detection_threshold,
                classification_threshold=inference_args.classification_threshold,
            )
            comb_target_fp, _ = combine_fwd_bck_preds(
                args.experiment_output_dir,
                fn,
                comb_iou_threshold=inference_args.comb_iou_threshold,
                comb_discard_threshold=inference_args.comb_discard_thresh,
                det_thresh=inference_args.detection_threshold,
            )
            print(f"Saving predictions for {fn} to {comb_target_fp}")

        else:
            print(f"Saving predictions for {fn} to {fwd_target_fp}")
