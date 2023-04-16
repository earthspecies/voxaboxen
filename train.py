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

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, args):

  model = model.to(device)
  n_anchors = len(args.anchor_durs_sec)
  pos_weight = torch.full([1], args.pos_weight, device = device) # default pos_weight = 1
  gamma = args.gamma # gamma = 0 means normal BCE loss
  class_loss_fn = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
  reg_loss_fn = masked_reg_loss
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad = True)
  
  train_evals = []
  test_evals = []
  
  val_dataloader = get_val_dataloader(args)
  
  for t in range(args.n_epochs):
      print(f"Epoch {t}\n-------------------------------")
      train_dataloader = get_train_dataloader(args, random_seed_shift = t) # reinitialize dataloader with different negatives each epoch
      model, train_eval = train_epoch(model, t, train_dataloader, class_loss_fn, reg_loss_fn, optimizer, args)
      test_eval = test_epoch(model, t, val_dataloader, class_loss_fn, reg_loss_fn, args)
      train_evals.append(train_eval.copy())
      test_evals.append(test_eval.copy())
      plot_eval(train_evals, test_evals, args)
  
  print("Done!")
  return model  
  
def train_epoch(model, t, dataloader, class_loss_fn, reg_loss_fn, optimizer, args):
    model.train()
    if t < args.unfreeze_encoder_epoch:
      model.freeze_encoder()
    else:
      model.unfreeze_encoder()
    
    
    evals = {}
    train_loss = 0; losses = []   
    data_iterator = tqdm.tqdm(dataloader)
    for i, (X, y, r, c) in enumerate(data_iterator):
      num_batches_seen = i
      X = torch.Tensor(X).to(device = device, dtype = torch.float)
            
      logits, regression = model(X)
      
      y = torch.Tensor(y).to(device = device, dtype = torch.float)
      r = torch.Tensor(r).to(device = device, dtype = torch.float)
      
      end_mask_perc = args.end_mask_perc
      end_mask_dur = int(logits.size(1)*end_mask_perc) 
      
      logits_flat = torch.reshape(logits[:,end_mask_dur:-end_mask_dur], (-1,1))
      y_flat = torch.reshape(y[:,end_mask_dur:-end_mask_dur], (-1, 1))
      
      class_loss = class_loss_fn(logits_flat, y_flat)
      reg_loss = reg_loss_fn(regression[:,end_mask_dur:-end_mask_dur,:], r[:,end_mask_dur:-end_mask_dur,:], y[:,end_mask_dur:-end_mask_dur])
      
      loss = class_loss + args.lamb* reg_loss
      train_loss += loss.item()
      losses.append(loss.item())
      
      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      
      optimizer.step()
      if i > 10:
        data_iterator.set_description(f"Loss {np.mean(losses[-10:]):.7f}")
    
    train_loss = train_loss / num_batches_seen
    evals['loss'] = float(train_loss)
    
    print(f"Epoch {t} | Train loss: {train_loss:1.3f}")
    return model, evals
                        
def test_epoch(model, t, dataloader, class_loss_fn, reg_loss_fn, args):
    model.eval()
    e = predict_and_evaluate(model, dataloader, args)
    
    summary = e['summary'][0.2]
    evals = {k:summary[k] for k in ['precision','recall','f1']}
    print(f"Epoch {t} | Test scores @0.5IoU: Precision: {evals['precision']:1.3f} Recall: {evals['recall']:1.3f} F1: {evals['f1']:1.3f}")
    return evals
    
def focal_loss(logits, y, pos_weight=1, gamma=0):
  # https://arxiv.org/pdf/1708.02002.pdf 
  if gamma==0:
    return F.binary_cross_entropy_with_logits(logits, y, reduction='mean', pos_weight=pos_weight)
  else:
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none', pos_weight=pos_weight)
    pt = torch.exp(-bce)
    fl = ((1-pt)**gamma)*bce
    return torch.mean(fl)
  
def masked_reg_loss(regression, r, y):
  # regression, r (Tensor): [batch, time, 2]
  # y (Tensor) : [batch, time], binary tensor of class probs
  
  reg_loss = F.mse_loss(regression, r, reduction='none')
  reg_loss = reg_loss * torch.unsqueeze(y,-1) # mask
  reg_loss = torch.sum(reg_loss, -1) # sum last dim
  reg_loss = torch.mean(reg_loss)
  return reg_loss