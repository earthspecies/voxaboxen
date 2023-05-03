from plumbum import local, FG
import os
import yaml
import numpy as np


def main():  
  local['python']['../main.py',
                  '--name=sweep0',
                  '--lr=0.003',
                  '--n-epochs=20',
                  '--clip-duration=16',
                  '--batch-size=8',
                  '--omit-empty-clip-prob=0.5',
                  '--clip-hop=8'] & FG
  
  local['python']['../main.py',
                  '--name=sweep1',
                  '--lr=0.001',
                  '--n-epochs=20',
                  '--clip-duration=16',
                  '--batch-size=8',
                  '--omit-empty-clip-prob=0.5',
                  '--clip-hop=8'] & FG
  
  local['python']['../main.py',
                  '--name=sweep2',
                  '--lr=0.001',
                  '--n-epochs=20',
                  '--clip-duration=4',
                  '--batch-size=8',
                  '--omit-empty-clip-prob=0.5',
                  '--clip-hop=2'] & FG
  
  local['python']['../main.py',
                  '--name=sweep3',
                  '--lr=0.001',
                  '--n-epochs=20',
                  '--clip-duration=16',
                  '--batch-size=8',
                  '--omit-empty-clip-prob=0.5',
                  '--clip-hop=8',
                  '--lamb=.02'] & FG
  
  evals = {}
  for i in range(4):
    name = f'sweep{i}'
    results_fp = f'/home/jupyter/sound_event_detection/logs/{name}/val_results/metrics.yaml'
    with open(results_fp, 'r') as f:
      results = yaml.safe_load(f)
    summary = results['summary'][0.8]
    e = []
    for l in summary:
        m = summary[l]['f1']
        e.append(m)
    evals[name] = float(np.mean(e))
    
  print(evals)             

if __name__ == "__main__":
  main()