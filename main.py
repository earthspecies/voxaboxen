from data import get_dataloader
from model import DetectionModel
from train import train
from util import parse_args, set_seed
from evaluation import generate_predictions, export_to_selection_table, get_metrics, summarize_metrics

import yaml
import sys
import os

def main(args):
  ## Setup
  args = parse_args(args)
  set_seed(args.seed)
  
  experiment_dir = os.path.join(args.output_dir, args.name)
  setattr(args, 'experiment_dir', str(experiment_dir))
  if not os.path.exists(args.experiment_dir):
    os.makedirs(args.experiment_dir)
  
  model = DetectionModel(args)
  dataloader = get_dataloader(args)
  
  ## Training
  trained_model = train(model, dataloader['train'], dataloader['val'], args)  
  
  ## Evaluation
  metrics = {}
  
  for fn in dataloader['test']:
    print(fn)
    predictions = generate_predictions(trained_model, dataloader['test'][fn], args)
    predictions_fp = export_to_selection_table(predictions, fn, args)
    annotations_fp = os.path.join(args.annotation_selection_tables_dir, f"{fn}.txt")
    metrics[fn] = get_metrics(predictions_fp, annotations_fp)
  
  summary = summarize_metrics(metrics)
  metrics['summary'] = summary
  
  metrics_fp = os.path.join(args.experiment_dir, 'metrics.yaml')
  with open(metrics_fp, 'w') as f:
    yaml.dump(metrics, f)
    

if __name__ == "__main__":
  main(sys.argv[1:])
  
  
# python main.py --name=debug --lr=0.001 --n-epochs=5
