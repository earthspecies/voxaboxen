import os
from shutil import copyfile
import argparse

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.config import CfgNode as CN

from voxaboxen.comparisons.dataloaders import collect_dataset_statistics

def get_full_cfg(sound_event_args, detectron_args):
    """ Combine command-line-arguments from sound event detection, detectron defaults, and custom detectron config
    See defaults: https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py
    See usage: https://github.com/facebookresearch/detectron2/blob/57bdb21249d5418c130d54e2ebdc94dda7a4c01a/docs/tutorials/configs.md
    See example config: https://github.com/facebookresearch/detectron2/blob/57bdb21249d5418c130d54e2ebdc94dda7a4c01a/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
    """

    # Create a CfgNode that fits the sound_event_detection and custom parameters
    cfg = get_cfg()    
    # New Spectrogram node with defaults
    cfg.SPECTROGRAM = CN()
    cfg.SPECTROGRAM.N_FFT = 400
    cfg.SPECTROGRAM.WIN_LENGTH = 400
    cfg.SPECTROGRAM.HOP_LENGTH = 200 #win_length // 2
    cfg.SPECTROGRAM.N_MELS = 64
    cfg.SPECTROGRAM.F_MIN = 20.
    cfg.SPECTROGRAM.REF = 1e-10
    cfg.SPECTROGRAM.FLOOR_THRESHOLD = 0.
    cfg.SPECTROGRAM.CEIL_THRESHOLD = 300.0
    # New Sound Event node
    cfg.SOUND_EVENT = CN(vars(sound_event_args))
    # New Evaluation node
    cfg.SOUND_EVENT.EVAL = CN()
    cfg.SOUND_EVENT.EVAL.TIME_BASED_NMS = False
    cfg.SOUND_EVENT.EVAL.IGNORE_INTERCLASS_IOU = False

    ## Some relevant detectron settings
    cfg.DATALOADER.NUM_WORKERS = -1 # Redundant with usual sound_event_detection args
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True # Redundant, will instead use value of --omit-empty-clip-prob
    
    # See https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md to choose models
    # These config files often have pre-trained weights included. To train from scratch, use `--opts MODEL.WEIGHTS ""`
    cfg.merge_from_file(model_zoo.get_config_file(detectron_args.detectron_base_config))
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.MASK_ON = False #We do not use any masks, only bounding boxes.
    cfg.MODEL.PIXEL_MEAN = [-1, 0.0, 0.0] # First element will be automatically updated based on train set statistics
    cfg.MODEL.PIXEL_STD = [-1, 1.0, 1.0]  # First element will be automatically updated based on train set statistics
    cfg.MODEL.BACKBONE.FREEZE_AT = 0 #For audio, we may want to retrain earliest layers?
    
    ## If you want these ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS to be different than in detectron_base_config, then:
    # 1. You can try automatic setting with --detectron-use-box-statistics (may not be stable if boxes are small)
    # 2. Or, use --detectron-config-fp for specifying them (rather than command line - see parse_args)
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4,20,100]] # Will be automatically updated based on train set statistics 
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1,1.0,5.0]] # Will be automatically updated based on train set statistics 

    cfg.MODEL.ROI_HEADS.NUM_CLASSES= 0 #Will be automatically set to correct number of classes based on project config
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST= 0.05 #See https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST= 0.5
    
    #For audio, do not have resizing or flipping
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.INPUT.MIN_SIZE_TRAIN = (0,) #Set to zero if no resizing. https://github.com/facebookresearch/detectron2/blob/dc4897d4d2ca1df7b922720186e481ccc7ba36a6/detectron2/data/transforms/augmentation_impl.py#L158
    cfg.INPUT.MAX_SIZE_TRAIN = 0 
    cfg.INPUT.MIN_SIZE_TEST = 0 #Set to zero to disable resize in testing.
    cfg.INPUT.MAX_SIZE_TEST = 0
    
    cfg.TEST.EVAL_PERIOD = 5000 # Test every 1000 steps

    cfg.SOLVER.IMS_PER_BATCH = sound_event_args.batch_size
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000 
    cfg.SOLVER.MAX_ITER = 100000

    # Add in Detectron custom parameters, either by config file or by command line list
    if detectron_args.detectron_config_fp is not None:
        cfg.merge_from_file(detectron_args.detectron_config_fp)
    if detectron_args.opts is not None:
        cfg.merge_from_list(detectron_args.opts)

    # Update specific detectron params based on sound_event params
    cfg.OUTPUT_DIR = cfg.SOUND_EVENT.experiment_output_dir
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg.SOUND_EVENT.label_set)
    cfg.DATASETS.TRAIN = (sound_event_args.train_info_fp,)
    cfg.DATASETS.TEST = (sound_event_args.val_info_fp,)

    # Update detectron params based on dataset
    collect_dataset_statistics(cfg, use_box_statistics=detectron_args.detectron_use_box_statistics); 
    
    # Save a copy of all parameters
    save_all_params(cfg)

    return cfg

def save_all_params(cfg):
    """ Save a copy of the params used for this experiment """
    with open(cfg.SOUND_EVENT.experiment_dir + "/all_params.yaml", "w") as f:
        f.write(cfg.dump())

def parse_args(args):
    """ Separate parser for detectron based command line args """
    parser = argparse.ArgumentParser()
  
    # General
    # To see available: https://github.com/facebookresearch/detectron2/tree/57bdb21249d5418c130d54e2ebdc94dda7a4c01a/configs
    parser.add_argument('--detectron-base-config', type = str, default="./COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", help="Base config that will be merged in early.")
    parser.add_argument('--detectron-use-box-statistics', action="store_true", help="Whether to decide anchor sizes and aspect ratio based on statistics of boxes in training set.")
    # If you want to change cfg.MODEL.ANCHOR_GENERATOR.SIZES or cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS, recommend to use --detectron-config-fp instead of --ops (hard to specify list of lists as PATH.KEY value pairs in command line.)
    parser.add_argument('--detectron-config-fp', type = str, required=False, help="If you prefer to indicate a config file for your custom detectron args, use this to point to the custom file.")
    # From https://github.com/facebookresearch/detectron2/blob/57bdb21249d5418c130d54e2ebdc94dda7a4c01a/detectron2/engine/defaults.py#L134
    # For how to use opts, see https://github.com/facebookresearch/detectron2/blob/57bdb21249d5418c130d54e2ebdc94dda7a4c01a/docs/tutorials/configs.md
    parser.add_argument(
        "--opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.".
                """.strip(),
                default=None,
                nargs=argparse.REMAINDER,
            )
    
    args = parser.parse_args(args)
    
    return args