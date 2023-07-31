import os
from shutil import copyfile

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.config import CfgNode as CN

from dataloaders import collect_dataset_statistics

def get_full_cfg(sound_event_args, detectron_config_fp):
    """ Combine command-line-arguments from sound event detection, detectron defaults, and custom detectron config"""

    # Get the correct detectron config
    experiment_dir = sound_event_args.experiment_dir
    saved_detectron_config_fp = os.path.join(experiment_dir, "detectron_cfg.yaml")
    detectron_config_fp = detectron_config_fp if not os.path.exists(saved_detectron_config_fp) else saved_detectron_config_fp
    if detectron_config_fp != saved_detectron_config_fp:
        copyfile(detectron_config_fp, saved_detectron_config_fp)

    # Create a CfgNode that fits the sound_event_detection and custom parameters
    cfg = get_cfg()
    # New Spectrogram node with defaults
    cfg.SPECTROGRAM = CN()
    cfg.SPECTROGRAM.N_FFT = 400
    cfg.SPECTROGRAM.WIN_LENGTH = 400
    cfg.SPECTROGRAM.HOP_LENGTH = 200 #win_length // 2
    cfg.SPECTROGRAM.N_MELS = 100
    cfg.SPECTROGRAM.F_MIN = 20.
    cfg.SPECTROGRAM.REF = 1e-8
    cfg.SPECTROGRAM.FLOOR_THRESHOLD = 0.
    cfg.SPECTROGRAM.CEIL_THRESHOLD = 120.0
    # New Sound Event node
    cfg.SOUND_EVENT = CN(vars(sound_event_args))

    # Add in detectron defaults, then detectron custom
    cfg.merge_from_file(model_zoo.get_config_file("./Base-RCNN-FPN.yaml"))
    cfg.merge_from_file(detectron_config_fp)

    # Update detectron params based on sound_event params
    cfg.OUTPUT_DIR = cfg.SOUND_EVENT.experiment_output_dir
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg.SOUND_EVENT.label_set)

    # Update detectron params based on dataset
    collect_dataset_statistics(cfg); 
    
    # Save a copy of all parameters
    save_all_params(cfg)

    return cfg

def save_all_params(cfg):
    """ Save a copy of the params used for this experiment """
    with open(cfg.SOUND_EVENT.experiment_dir + "/all_params.yaml", "w") as f:
        f.write(cfg.dump())