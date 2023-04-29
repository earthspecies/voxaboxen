import numpy as np
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from plotters import plot_eval
from evaluation import predict_and_evaluate
from functools import partial
from data import get_train_dataloader, get_val_dataloader
from model import preprocess_and_augment
import os

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, args):
  model = model.to(device)
  class_loss_fn = modified_focal_loss
  reg_loss_fn = masked_reg_loss
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad = True)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, eta_min=args.lr/100, last_epoch=- 1, verbose=False)
  
  train_evals = []
  test_evals = []
  learning_rates = []
  
  val_dataloader = get_val_dataloader(args)
      
  for t in range(args.n_epochs):
      print(f"Epoch {t}\n-------------------------------")
      train_dataloader = get_train_dataloader(args, random_seed_shift = t) # reinitialize dataloader with different negatives each epoch
      model, train_eval = train_epoch(model, t, train_dataloader, class_loss_fn, reg_loss_fn, optimizer, args)
      test_eval = test_epoch(model, t, val_dataloader, class_loss_fn, reg_loss_fn, args)
      train_evals.append(train_eval.copy())
      test_evals.append(test_eval.copy())
      learning_rates.append(optimizer.param_groups[0]["lr"])
      plot_eval(train_evals, test_evals, learning_rates, args)
      scheduler.step()

  print("Done!")
  
  checkpoint_dict = {
        "epoch": t,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_evals": train_evals,
        "test_evals" : test_evals
    }

  torch.save(
      checkpoint_dict,
      os.path.join(args.experiment_dir, f"final_model.pt"),
  ) 
  
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
      X = torch.Tensor(X).to(device = device, dtype = torch.float)
      y = torch.Tensor(y).to(device = device, dtype = torch.float)
      r = torch.Tensor(r).to(device = device, dtype = torch.float)
      loss_mask = torch.Tensor(loss_mask).to(device = device, dtype = torch.float)
      
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
                        
def test_epoch(model, t, dataloader, class_loss_fn, reg_loss_fn, args):
    model.eval()
    e, _ = predict_and_evaluate(model, dataloader, args, save = False)
    
    summary = e['summary'][0.5]
    
    evals = {k:[] for k in ['precision','recall','f1']}
    for k in ['precision','recall','f1']:
      for l in args.label_set:
        m = summary[l][k]
        evals[k].append(m)
      evals[k] = float(np.mean(evals[k]))
        
    print(f"Epoch {t} | Test scores @0.5IoU: Precision: {evals['precision']:1.3f} Recall: {evals['recall']:1.3f} F1: {evals['f1']:1.3f}")
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
    
#   num_pos  = pos_inds.float().sum()
#   pos_loss = pos_loss.sum()
#   neg_loss = neg_loss.sum()
    
#   # print(f"pos {-1.*pos_loss} neg {-1.*neg_loss}")

#   if num_pos == 0:
#     loss = loss - neg_loss
#   else:
#     loss = loss - (pos_loss + neg_loss) / num_pos
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
