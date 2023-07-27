import os
from shutil import copyfile

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.config import CfgNode as CN

# Defaults go here 
def get_sound_event_cfg(sound_event_args, detectron_config_fp):
    cfg = get_cfg()

    cfg.SPECTROGRAM = CN()
    cfg.SPECTROGRAM.N_FFT = 400
    cfg.SPECTROGRAM.WIN_LENGTH = 400
    cfg.SPECTROGRAM.HOP_LENGTH = 200 #win_length // 2
    cfg.SPECTROGRAM.N_MELS = 100
    cfg.SPECTROGRAM.REF = 1e-8
    cfg.SPECTROGRAM.FLOOR_THRESHOLD = 0.
    cfg.SPECTROGRAM.CEIL_THRESHOLD = 120.0

    cfg.SOUND_EVENT = CN(vars(sound_event_args))

    cfg.merge_from_file(model_zoo.get_config_file("./Base-RCNN-FPN.yaml"))
    cfg.merge_from_file(detectron_config_fp)

    cfg.OUTPUT_DIR = cfg.SOUND_EVENT.experiment_output_dir
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg.SOUND_EVENT.label_set)

    return cfg

def save_custom_detectron_params(detectron_config_fp, args):
    """ Save a copy of the params used for this experiment """
    params_file = os.path.join(args.experiment_dir, "detectron_cfg.yaml")
    copyfile(detectron_config_fp, params_file)
