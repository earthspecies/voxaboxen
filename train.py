import numpy as np
import torch
from torch import nn
import torch.nn as nn
import tqdm
from plotters import plot_eval
from evaluation import predict_and_evaluate

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, train_dataloader, test_dataloader, args):
  model = model.to(device)
  n_anchors = len(args.anchor_durs_sec)
  pos_weight = torch.full([n_anchors], args.pos_weight, device = device) # default pos_weight = 1
  loss_fn = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
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
    e = predict_and_evaluate(model, dataloader, args)['summary'][0.5]
    evals = {k:e[k] for k in ['precision','recall','f1']}
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