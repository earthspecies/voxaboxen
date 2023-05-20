import numpy as np
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from functools import partial
import os

from source.evaluation.plotters import plot_eval
from source.evaluation.evaluation import predict_and_evaluate
from source.data.data import get_train_dataloader, get_val_dataloader
from source.model.model import preprocess_and_augment

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, args):
  model = model.to(device)
  class_loss_fn = modified_focal_loss
  reg_loss_fn = masked_reg_loss
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad = True)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=0.1, last_epoch=- 1, verbose=False)
  # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, eta_min=args.lr/100, last_epoch=- 1, verbose=False)
  
  train_evals = []
  learning_rates = []
  val_evals = []
  
  if args.early_stopping:
    assert args.val_info_fp is not None
    best_f1 = 0

  if args.val_info_fp is not None:
    val_dataloader = get_val_dataloader(args)
    use_val = True
  else:
    use_val = False
      
  for t in range(args.n_epochs):
      print(f"Epoch {t}\n-------------------------------")
      train_dataloader = get_train_dataloader(args, random_seed_shift = t) # reinitialize dataloader with different negatives each epoch
      model, train_eval = train_epoch(model, t, train_dataloader, class_loss_fn, reg_loss_fn, optimizer, args)
      train_evals.append(train_eval.copy())
      learning_rates.append(optimizer.param_groups[0]["lr"])
      if use_val:
        val_eval = val_epoch(model, t, val_dataloader, class_loss_fn, reg_loss_fn, args)
        val_evals.append(val_eval.copy())
        plot_eval(train_evals, learning_rates, args, test_evals = val_evals)
      else:
        plot_eval(train_evals, learning_rates, args)
      scheduler.step()
      
      if use_val and args.early_stopping:
        current_f1 = val_eval['f1']
        if current_f1 > best_f1:
          print('found new best model')
          best_f1 = current_f1

          checkpoint_dict = {
          "epoch": t,
          "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
          "scheduler_state_dict": scheduler.state_dict(),
          "train_evals": train_evals,
          "val_evals" : val_evals
          }
          
          torch.save(
              checkpoint_dict,
              os.path.join(args.experiment_dir, f"model.pt"),
          ) 
          
      else:  
        checkpoint_dict = {
        "epoch": t,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_evals": train_evals,
        "val_evals" : val_evals
        }
        
        torch.save(
              checkpoint_dict,
              os.path.join(args.experiment_dir, f"model.pt"),
          ) 
          
  
  print("Done!")
  
  cp = torch.load(os.path.join(args.experiment_dir, f"model.pt"))
  model.load_state_dict(cp["model_state_dict"])
  
  # resave validation with best model
  if use_val:
    val_epoch(model, t+1, val_dataloader, class_loss_fn, reg_loss_fn, args)
  
  return model  
  
def train_epoch(model, t, dataloader, class_loss_fn, reg_loss_fn, optimizer, args):
    model.train()
    if t < args.unfreeze_encoder_epoch:
      model.freeze_encoder()
    else:
      model.unfreeze_encoder()
    
    
    evals = {}
    train_loss = 0; losses = []; detection_losses = []; regression_losses = []
    data_iterator = tqdm.tqdm(dataloader)
    for i, (X, y, r, loss_mask) in enumerate(data_iterator):
      num_batches_seen = i
      X = X.to(device = device, dtype = torch.float)
      y = y.to(device = device, dtype = torch.float)
      r = r.to(device = device, dtype = torch.float)
      loss_mask = loss_mask.to(device = device, dtype = torch.float)
      
      X, y, r, loss_mask = preprocess_and_augment(X, y, r, loss_mask, True, args)
      probs, regression = model(X)
      
      end_mask_perc = args.end_mask_perc
      end_mask_dur = int(probs.size(1)*end_mask_perc) 
      
      probs_clipped = probs[:,end_mask_dur:-end_mask_dur,:]
      y_clipped = y[:,end_mask_dur:-end_mask_dur,:]
      regression_clipped = regression[:,end_mask_dur:-end_mask_dur,:]
      r_clipped = r[:,end_mask_dur:-end_mask_dur,:]
      loss_mask_clipped = loss_mask[:, end_mask_dur:-end_mask_dur]
      
      class_loss = class_loss_fn(probs_clipped, y_clipped, mask=loss_mask_clipped)
      reg_loss = reg_loss_fn(regression_clipped, r_clipped, y_clipped, mask=loss_mask_clipped)
      
      loss = class_loss + args.lamb* reg_loss
      train_loss += loss.item()
      losses.append(loss.item())
      detection_losses.append(class_loss.item())
      regression_losses.append(args.lamb * reg_loss.item())
      
      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      
      optimizer.step()
      if i > 10:
        data_iterator.set_description(f"Loss {np.mean(losses[-10:]):.7f}, Detection Loss {np.mean(detection_losses[-10:]):.7f}, Regression Loss {np.mean(regression_losses[-10:]):.7f}")
    
    train_loss = train_loss / num_batches_seen
    evals['loss'] = float(train_loss)
    
    print(f"Epoch {t} | Train loss: {train_loss:1.3f}")
    return model, evals
                        
def val_epoch(model, t, dataloader, class_loss_fn, reg_loss_fn, args):
    model.eval()
    e, _ = predict_and_evaluate(model, dataloader, args, output_dir = os.path.join(args.experiment_dir, 'val_results'), verbose = False)
    
    summary = e['summary'][args.model_selection_iou]
    
    evals = {k:[] for k in ['precision','recall','f1']}
    for k in ['precision','recall','f1']:
      for l in args.label_set:
        m = summary[l][k]
        evals[k].append(m)
      evals[k] = float(np.mean(evals[k]))
        
    print(f"Epoch {t} | Test scores @{args.model_selection_iou}IoU: Precision: {evals['precision']:1.3f} Recall: {evals['recall']:1.3f} F1: {evals['f1']:1.3f}")
    return evals

def modified_focal_loss(pred, gt, mask = None):
  # Modified from https://github.com/xingyizhou/CenterNet/blob/2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/losses.py
  ''' 
      pred [batch, time, n_classes]
      gt [batch, time, n_classes]
      mask (Tensor) : [batch, time], binary tensor
  '''
  
  n_classes = pred.size(-1)  
  
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
  
  loss = -1.*n_classes*(neg_loss + pos_loss)
  
  if mask is not None:
    loss = loss * mask.unsqueeze(-1)
  
  loss = loss.mean()
  return loss
  
  
def masked_reg_loss(regression, r, y, mask = None):
  # regression, r (Tensor): [batch, time, n_classes]
  # y (Tensor) : [batch, time, n_classes], float tensor
  # mask (Tensor) : [batch, time], binary tensor
  
  reg_loss = F.l1_loss(regression, r, reduction='none')
  if mask is None:
    mask = y.eq(1).float()
  else:
    mask = mask.unsqueeze(-1) * y.eq(1).float()
  reg_loss = reg_loss * mask
  reg_loss = torch.sum(reg_loss)
  n_pos = mask.sum()
  
  if n_pos>0:
    reg_loss = reg_loss / n_pos
    
  return reg_loss
