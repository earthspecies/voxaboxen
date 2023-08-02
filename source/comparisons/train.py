from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import sys
from glob import glob

#Import custom code
from source.comparisons.dataloaders import SoundEventTrainer
from source.comparisons.params import get_full_cfg, parse_args
from source.training.train import train
import source.training.params as aves_params

def train(args):

    # Standard args
    sound_event_args, detectron_args = aves_params.parse_args(args, allow_unknown=True)
    detectron_args = parse_args(detectron_args)
    aves_params.set_seed(sound_event_args.seed)

    experiment_dir = os.path.join(sound_event_args.project_dir, sound_event_args.name)
    setattr(sound_event_args, 'experiment_dir', str(experiment_dir))
    if not os.path.exists(sound_event_args.experiment_dir):
        os.makedirs(sound_event_args.experiment_dir)

    experiment_output_dir = os.path.join(experiment_dir, "outputs")
    setattr(sound_event_args, 'experiment_output_dir', experiment_output_dir)
    if not os.path.exists(sound_event_args.experiment_output_dir):
        os.makedirs(sound_event_args.experiment_output_dir)
    aves_params.save_params(sound_event_args)

    # Detectron config
    cfg = get_full_cfg(sound_event_args, detectron_args)
    
    # ~~~~ Train
    print("Existing model checkpoints:", glob(cfg.OUTPUT_DIR + "/*.pth"))
    n_ckpts = len(glob(cfg.OUTPUT_DIR + "/*.pth"))
    resume = True if n_ckpts > 0 else False
    trainer = SoundEventTrainer(cfg)
    trainer.resume_or_load(resume=resume) 
    print("Let's train~", flush=True)
    trainer.train()

    # ~~~~ Evaluate on test set
    # cfg.SOUND_EVENT.experiment_dir + "/all_params.yaml"
    # TODO (high priority) automatically call evaluate.py

if __name__ == "__main__":
    train(sys.argv[1:])

