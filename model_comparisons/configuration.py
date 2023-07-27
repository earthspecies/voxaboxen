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
    cfg.SPECTROGRAM.N_MELS = 128
    cfg.SPECTROGRAM.FLOOR_THRESHOLD = 0.
    cfg.SPECTROGRAM.CEIL_THRESHOLD = 120.0

    #TODO: Don't know if this will work.
    cfg.SOUND_EVENT = CN()
    cfg.SOUND_EVENT.ARGS = sound_event_args

    cfg.merge_from_file(model_zoo.get_config_file("./Base-RCNN-FPN.yaml"))
    cfg.merge_from_file(detectron_config_fp)

    return cfg

def save_custom_detectron_params(detectron_config_fp, args):
    """ Save a copy of the params used for this experiment """
    params_file = os.path.join(args.experiment_dir, "detectron_cfg.yaml")
    copyfile(detectron_config_fp, params_file)
