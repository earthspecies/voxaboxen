import sys
sys.path.append("/home/jupyter/sound_event_detection/")
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import sys
from glob import glob

#Import custom code
from dataloaders import SoundEventTrainer
from configuration import get_sound_event_cfg, save_custom_detectron_params
from source.training.train import train
from source.training.params import parse_args, set_seed, save_params

def train(dataset_name, detectron_config_fp, args):

    # Standard args
    sound_event_args = parse_args(args)
    set_seed(sound_event_args.seed)

    experiment_dir = os.path.join(sound_event_args.project_dir, sound_event_args.name)
    setattr(sound_event_args, 'experiment_dir', str(experiment_dir))
    if not os.path.exists(sound_event_args.experiment_dir):
        os.makedirs(sound_event_args.experiment_dir)

    experiment_output_dir = os.path.join(experiment_dir, "outputs")
    setattr(sound_event_args, 'experiment_output_dir', experiment_output_dir)
    if not os.path.exists(sound_event_args.experiment_output_dir):
        os.makedirs(sound_event_args.experiment_output_dir)
    save_params(sound_event_args)

    # Detectron config
    cfg = get_sound_event_cfg(sound_event_args, detectron_config_fp)
    save_custom_detectron_params(detectron_config_fp, sound_event_args)

    # ~~~~
    print("Existing model checkpoints:", glob(cfg.OUTPUT_DIR + "*.pth"))
    n_ckpts = len(glob(cfg.OUTPUT_DIR + "*.pth"))
    resume = True if n_ckpts > 0 else False
    cfg.DATASETS.TRAIN = dataset_name
    cfg.DATASETS.TEST = dataset_name

    trainer = SoundEventTrainer(cfg)
    trainer.resume_or_load(resume=resume) 
    print("Let's train", flush=True)
    trainer.train()

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], sys.argv[3:])

