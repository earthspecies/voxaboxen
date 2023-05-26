import os
import sys
import pandas as pd
import torch

from source.inference.params import parse_inference_args
from source.training.params import load_params
from source.model.model import DetectionModel
from source.evaluation.evaluation import generate_predictions, export_to_selection_table
from source.data.data import get_single_clip_data

device = "cuda" if torch.cuda.is_available() else "cpu"

def inference(inference_args):
  inference_args = parse_inference_args(inference_args)
  args = load_params(inference_args.model_args_fp)  
  
  model = DetectionModel(args)
  
  # model  
  model_checkpoint_fp = os.path.join(args.experiment_dir, "model.pt")
  print(f"Loading model weights from {model_checkpoint_fp}")
  cp = torch.load(model_checkpoint_fp)
  model.load_state_dict(cp["model_state_dict"])
  model = model.to(device)
  
  
  
  files_to_infer = pd.read_csv(inference_args.file_info_for_inference)
  
  for i, row in files_to_infer.iterrows():
    audio_fp = row['audio_fp']
    fn = row['fn']
    
    if not os.path.exists(audio_fp):
      print(f"Could not locate file {audio_fp}")
      continue
    
    dataloader = get_single_clip_data(audio_fp, args.clip_duration/2, args)
    predictions, regressions = generate_predictions(model, dataloader, args, verbose=True)
    predictions_fp = export_to_selection_table(predictions, regressions, fn, args, verbose = True)
    
    print(f"Saving predictions for {fn} to {predictions_fp}")

if __name__ == "__main__":
    main(sys.argv[1:])

