import sys
import os
import yaml
import pandas as pd
import torch
from time import time
from voxaboxen.training.params import parse_args, set_seed, save_params
from voxaboxen.evaluation.evaluation import predict_and_generate_manifest, combine_fwd_bck_preds
from voxaboxen.model.model import DetectionModel
from voxaboxen.data.data import get_test_dataloader, get_val_dataloader


args = parse_args(sys.argv[1:])

set_seed(args.seed)

experiment_dir = os.path.join(args.project_dir, args.name)

if os.path.exists(os.path.join(experiment_dir, 'model.pt')):
    if args.exists_strategy=='resume':
        args.previous_checkpoint_fp = os.path.join(experiment_dir, 'model.pt')
        with open(os.path.join(experiment_dir, 'train_history.yaml')) as f:
            x = yaml.load(f, Loader=yaml.SafeLoader)
        n_epochs_ran_for = len(x)
        args.n_epochs -= n_epochs_ran_for
        print(f'resuming previous run which ran for {n_epochs_ran_for} epochs, now training for the remaining {args.n_epochs}')
        assert max(x.keys()) == n_epochs_ran_for-1
        args.unfreeze_encoder_epoch = max(0, args.unfreeze_encoder_epoch-n_epochs_ran_for)

    elif args.exists_strategy == 'none' and args.name!='demo':
        sys.exit('experiment already exists with this name')

experiment_output_dir = os.path.join(experiment_dir, "outputs")
if not os.path.exists(experiment_output_dir):
    os.makedirs(experiment_output_dir)

setattr(args, 'experiment_dir', str(experiment_dir))
setattr(args, 'experiment_output_dir', experiment_output_dir)
save_params(args)
model = DetectionModel(args).to(args.device)

if args.previous_checkpoint_fp is not None:
    print(f"loading model weights from {args.previous_checkpoint_fp}")
    cp = torch.load(args.previous_checkpoint_fp)
    if "model_state_dict" in cp.keys():
        cp = cp["model_state_dict"]
    if not args.bidirectional:
        cp = {k:v for k,v in cp.items() if not k.startswith('rev_detection_head')}
    #cp = {k.replace('encoder.beats', 'encoder.model'):v for k,v in cp.items()}
    model.load_state_dict(cp)

best_pred_type = 'comb' if args.bidirectional else 'fwd'
#split_info_fps = []
#for split in ['train', 'val', 'test']:
#    fp = getattr(args, f'{split}_info_fp')
#    df = pd.read_csv(fp)
#    split_info_fps.append(df)
#combined_info_df = pd.concat(split_info_fps, axis=0)
#combined_info_df.to_csv('tmp.csv')
#args.test_info_fp = 'tmp.csv'
dloader = get_test_dataloader(args)
starttime = time()
manifest = predict_and_generate_manifest(model, dloader, args, verbose=False)[0.5]
if args.bidirectional:
    for i, row in manifest.iterrows():
        fn = row['filename']
        annots_fp = row['annotations_fp']
        duration = row['duration_sec']
        row['comb_predictions_fp'], row['match_predictions_fp'] = combine_fwd_bck_preds(args.experiment_output_dir, fn, comb_discard_threshold=0, comb_iou_thresh=0.5, det_thresh=args.detection_threshold)
inference_time = time() - starttime
print(f'Inference time: {inference_time:.5f}')
dset = args.project_config_fp.split(os.path.sep)[1].removesuffix('_experiment')
os.makedirs('inference_times', exist_ok=True)
with open(f'inference_times/{dset}-{args.device}-bid{args.bidirectional}.txt', 'w') as f:
    f.write(str(inference_time))

