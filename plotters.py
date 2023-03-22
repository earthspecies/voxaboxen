from matplotlib import pyplot as plt
from pathlib import Path

def plot_eval(train_evals, test_evals, args):
  # train_evals : list of dicts, one dict per epoch
  # test_evals : list of dicts, one dict per epoch
  eval_output_dir = Path(args.output_dir)
  plot_fp = Path(eval_output_dir, "train_progress.png")
  train_keys = train_evals[0].keys()
  test_keys = test_evals[0].keys()
  
  n_plots = len(train_keys)+len(test_keys)
  fig, ax = plt.subplots(nrows=n_plots, sharex=True)
  
  plot_number = 0
  for i, eval_dict_list in enumerate([train_evals, test_evals]):
    fold = {0:"Train", 1:"Test"}[i]
    for key in sorted(eval_dict_list[0].keys()):
      toplot = [d[key] for d in eval_dict_list]
      ax[plot_number].plot(toplot)
      ax[plot_number].set_title(f"{fold} {key}")
      ax[plot_number].set_xlabel("Epoch")
      plot_number += 1
      
  plt.savefig(plot_fp)
    
  