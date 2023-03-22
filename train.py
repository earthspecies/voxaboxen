import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, train_dataloader, test_dataloader, args):
  model = model.to(device)
  n_anchors = len(args.anchor_durs_sec)
  pos_weight = torch.full([n_anchors], args.pos_weight)
  loss_fn = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad = True)

  for t in range(args.n_epochs):
      print(f"Epoch {t}\n-------------------------------")
      model = train_epoch(model, t, train_dataloader, loss_fn, optimizer, args)
      # test_epoch(model, t, test_dataloader, loss_fn, args)
  
  print("Done!")
  return model  
  
def train_epoch(model, t, dataloader, loss_fn, optimizer, args):
    model.train()
    train_loss = 0; losses = []   
    data_iterator = tqdm.tqdm(dataloader)
    for i, (X, y) in enumerate(data_iterator):
      num_batches_seen = i
      X = torch.Tensor(X).to(device = device, dtype = torch.float)
      logits = model(X)
      logits = torch.reshape(logits, (-1, logits.size(-1)))
      y = torch.reshape(y, (-1, y.size(-1))
      loss = loss_fn(X, y)
      train_loss += loss.item()
      losses.append(loss.item())
      
      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      
      optimizer.step()
      if i > 10:
        data_iterator.set_description(f"Loss {np.mean(losses[-10:]):.7f}")
      if i % 50:
        plt.plot(losses)
        plt.savefig("/home/jupyter/loss.png")
        plt.close()
    
    train_loss = train_loss / num_batches_seen
    print(f"Epoch {t} | Train loss: {train_loss:1.3f}")
    return model