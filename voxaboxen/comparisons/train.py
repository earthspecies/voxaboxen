from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import sys
from glob import glob

#Import custom code
from voxaboxen.comparisons.dataloaders import SoundEventTrainer
from voxaboxen.comparisons.params import get_full_cfg, parse_args
from voxaboxen.comparisons.evaluate import run_evaluation
from voxaboxen.training.train import train
import voxaboxen.training.params as aves_params

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
    try:
        trainer.train()
    except StopIteration:
        print("Reached stop iteration. Training complete.")

    # ~~~~ Evaluate on val and test set
    print("Running evaluation on train set",flush=True)
    run_evaluation(trainer.model, cfg.SOUND_EVENT.train_info_fp, cfg, "train_results")
    if cfg.SOUND_EVENT.val_info_fp is not None:
        print("Running evaluation on val set",flush=True)
        run_evaluation(trainer.model, cfg.SOUND_EVENT.val_info_fp, cfg, "val_results")
    if cfg.SOUND_EVENT.test_info_fp is not None:
        print("Running evaluation on test set",flush=True)
        run_evaluation(trainer.model, cfg.SOUND_EVENT.test_info_fp, cfg, "test_results")

if __name__ == "__main__":
    train(sys.argv[1:])

