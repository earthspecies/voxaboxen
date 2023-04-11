import numpy as np
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from plotters import plot_eval
from evaluation import predict_and_evaluate
from functools import partial

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, train_dataloader, test_dataloader, args):
  model = model.to(device)
  n_anchors = len(args.anchor_durs_sec)
  pos_weight = torch.full([n_anchors], args.pos_weight, device = device) # default pos_weight = 1
  gamma = args.gamma # gamma = 0 means normal BCE loss
  loss_fn = partial(focal_loss, pos_weight=pos_weight, gamma=gamma) # nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad = True)
  
  train_evals = []
  test_evals = []
  
  for t in range(args.n_epochs):
      print(f"Epoch {t}\n-------------------------------")
      model, train_eval = train_epoch(model, t, train_dataloader, loss_fn, optimizer, args)
      test_eval = test_epoch(model, t, test_dataloader, loss_fn, args)
      train_evals.append(train_eval.copy())
      test_evals.append(test_eval.copy())
      plot_eval(train_evals, test_evals, args)
  
  print("Done!")
  return model  
  
def train_epoch(model, t, dataloader, loss_fn, optimizer, args):
    model.train()
    if t < args.unfreeze_encoder_epoch:
      model.freeze_encoder()
    else:
      model.unfreeze_encoder()
    
    
    evals = {}
    train_loss = 0; losses = []   
    data_iterator = tqdm.tqdm(dataloader)
    for i, (X, y, c) in enumerate(data_iterator):
      num_batches_seen = i
      X = torch.Tensor(X).to(device = device, dtype = torch.float)

      logits = model(X)
      
      # aves may have a 1 sample difference from targets
      y = torch.Tensor(y).to(device = device, dtype = torch.float)
      y = y[:,:logits.size(1),:]
      
      logits = torch.reshape(logits, (-1, logits.size(-1)))
      y = torch.reshape(y, (-1, y.size(-1)))
      loss = loss_fn(logits, y)
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
                        
def test_epoch(model, t, dataloader, loss_fn, args):
    model.eval()
    e = predict_and_evaluate(model, dataloader, args)
#     ## temporary
#     evals = {}
#     keys = list(e.keys())
#     from matplotlib import pyplot as plt
#     for key in keys:
#       counts = e[key][0.5]
      
#       tp = counts['TP']
#       fp = counts['FP']
#       fn = counts['FN']

#       if tp + fp == 0:
#         prec = 1
#       else:
#         prec = tp / (tp + fp)

#       if tp + fn == 0:
#         rec = 1
#       else:
#         rec = tp / (tp + fn)

#       if prec + rec == 0:
#         f1 = 0
#       else:
#         f1 = 2*prec*rec / (prec + rec)      
        
#       evals[key] = f1
      
#     ##
    
    summary = e['summary'][0.2]
    evals = {k:summary[k] for k in ['precision','recall','f1']}
    print(f"Epoch {t} | Test scores @0.5IoU: Precision: {evals['precision']:1.3f} Recall: {evals['recall']:1.3f} F1: {evals['f1']:1.3f}")
    return evals
    
    
    
#     evals = {}
#     test_loss = 0 
#     data_iterator = tqdm.tqdm(dataloader)
#     with torch.no_grad():
#       for i, (X, y, c) in enumerate(data_iterator):
#         num_batches_seen = i
#         X = torch.Tensor(X).to(device = device, dtype = torch.float)

#         logits = model(X)
#         y = torch.Tensor(y).to(device = device, dtype = torch.float)
#         y = y[:,:logits.size(1),:]
#         logits = torch.reshape(logits, (-1, logits.size(-1)))
#         y = torch.reshape(y, (-1, y.size(-1)))
#         loss = loss_fn(logits, y)
#         test_loss += loss.item()
    
#     test_loss = test_loss / num_batches_seen
#     evals['loss'] = float(test_loss)
    # print(f"Epoch {t} | Test loss: {test_loss:1.3f}")
    # return evals
    
def focal_loss(logits, y, pos_weight=1, gamma=0):
  # https://arxiv.org/pdf/1708.02002.pdf 
  if gamma==0:
    return F.binary_cross_entropy_with_logits(logits, y, reduction='mean', pos_weight=pos_weight)
  else:
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none', pos_weight=pos_weight)
    pt = torch.exp(-bce)
    fl = ((1-pt)**gamma)*bce
    return torch.mean(fl)