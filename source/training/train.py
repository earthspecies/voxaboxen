import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from functools import partial
import os
from einops import rearrange

from source.evaluation.plotters import plot_eval
from source.evaluation.evaluation import predict_and_generate_manifest, evaluate_based_on_manifest
from source.data.data import get_train_dataloader, get_val_dataloader
from source.model.model import rms_and_mixup

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
  import warnings
  warnings.warn("Only using CPU! Check CUDA")

def train(model, args):
  model = model.to(device)
  
  if args.previous_checkpoint_fp is not None:
    print(f"loading model weights from {args.previous_checkpoint_fp}")
    cp = torch.load(args.previous_checkpoint_fp)
    model.load_state_dict(cp["model_state_dict"])
  
  detection_loss_fn = modified_focal_loss
  reg_loss_fn = get_reg_loss_fn(args)
  
  class_loss_fn = get_class_loss_fn(args)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad = True)
  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=0.1, last_epoch=- 1, verbose=False)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, eta_min=0, last_epoch=- 1, verbose=False)
  
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
      model, train_eval = train_epoch(model, t, train_dataloader, detection_loss_fn, reg_loss_fn, class_loss_fn, optimizer, args)
      train_evals.append(train_eval.copy())
      learning_rates.append(optimizer.param_groups[0]["lr"])
      if use_val:
        val_eval = val_epoch(model, t, val_dataloader, detection_loss_fn, reg_loss_fn, class_loss_fn, args)
        val_evals.append(val_eval.copy())
        plot_eval(train_evals, learning_rates, args, val_evals = val_evals)
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
    val_epoch(model, t+1, val_dataloader, detection_loss_fn, reg_loss_fn, class_loss_fn, args)
  
  return model  
  
def train_epoch(model, t, dataloader, detection_loss_fn, reg_loss_fn, class_loss_fn, optimizer, args):
    model.train()
    if t < args.unfreeze_encoder_epoch:
      model.freeze_encoder()
    else:
      model.unfreeze_encoder()
    
    
    evals = {}
    train_loss = 0; losses = []; detection_losses = []; regression_losses = []; class_losses = []
    data_iterator = tqdm.tqdm(dataloader)
    for i, (X, d, r, y) in enumerate(data_iterator):
      num_batches_seen = i
      X = X.to(device = device, dtype = torch.float)
      d = d.to(device = device, dtype = torch.float)
      r = r.to(device = device, dtype = torch.float)
      y = y.to(device = device, dtype = torch.float)
      
      X, d, r, y = rms_and_mixup(X, d, r, y, True, args)
      probs, regression, class_logits = model(X)
      
      # We mask out loss from each end of the clip, so the model isn't forced to learn to detect events that are partially cut off.
      # This does not affect inference, because during inference we overlap clips at 50%
      
      end_mask_perc = args.end_mask_perc
      end_mask_dur = int(probs.size(1)*end_mask_perc) 
      
      d_clipped = d[:,end_mask_dur:-end_mask_dur]
      probs_clipped = probs[:,end_mask_dur:-end_mask_dur]
      
      regression_clipped = regression[:,end_mask_dur:-end_mask_dur]
      r_clipped = r[:,end_mask_dur:-end_mask_dur]
      
      class_logits_clipped = class_logits[:,end_mask_dur:-end_mask_dur,:]
      y_clipped = y[:,end_mask_dur:-end_mask_dur,:]
      
      
      detection_loss = detection_loss_fn(probs_clipped, d_clipped, pos_loss_weight = args.pos_loss_weight)
      reg_loss = reg_loss_fn(regression_clipped, r_clipped, d_clipped, y_clipped)
      class_loss = class_loss_fn(class_logits_clipped, y_clipped, d_clipped)
      
      loss = args.rho * class_loss + detection_loss + args.lamb * reg_loss
      train_loss += loss.item()
      losses.append(loss.item())
      detection_losses.append(detection_loss.item())
      regression_losses.append(args.lamb * reg_loss.item())
      class_losses.append(args.rho * class_loss.item())
      
      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      
      optimizer.step()
      if i > 10:
        data_iterator.set_description(f"Loss {np.mean(losses[-10:]):.7f}, Detection Loss {np.mean(detection_losses[-10:]):.7f}, Regression Loss {np.mean(regression_losses[-10:]):.7f}, Classification Loss {np.mean(class_losses[-10:]):.7f}")
    
    train_loss = train_loss / num_batches_seen
    evals['loss'] = float(train_loss)
    
    print(f"Epoch {t} | Train loss: {train_loss:1.3f}")
    return model, evals
                        
def val_epoch(model, t, dataloader, detection_loss_fn, reg_loss_fn, class_loss_fn, args):
    model.eval()
    
    manifest = predict_and_generate_manifest(model, dataloader, args, verbose = False)
    e, _ = evaluate_based_on_manifest(manifest, args, output_dir = os.path.join(args.experiment_dir, 'val_results'), iou = args.model_selection_iou, class_threshold = args.model_selection_class_threshold)
        
    summary = e['summary']
    
    evals = {k:[] for k in ['precision','recall','f1']}
    for k in ['precision','recall','f1']:
      for l in args.label_set:
        m = summary[l][k]
        evals[k].append(m)
      evals[k] = float(np.mean(evals[k]))
        
    print(f"Epoch {t} | Test scores @{args.model_selection_iou}IoU: Precision: {evals['precision']:1.3f} Recall: {evals['recall']:1.3f} F1: {evals['f1']:1.3f}")
    return evals

def modified_focal_loss(pred, gt, pos_loss_weight = 1):
  # Modified from https://github.com/xingyizhou/CenterNet/blob/2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/losses.py
  ''' 
      pred [batch, time,]
      gt [batch, time,]
  ''' 
  
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * pos_loss_weight
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
  
  loss = -1.*(neg_loss + pos_loss)
  
  loss = loss.mean()
  return loss
  
  
def masked_reg_loss(regression, r, d, y, class_weights = None):
  # regression, r (Tensor): [batch, time,]
  # r (Tensor) : [batch, time,], float tensor
  # d (Tensor) : [batch, time,], float tensor
  # y (Tensor) : [batch, time, n_classes]
  # class_weights (Tensor) : [n_classes,]
    
  reg_loss = F.l1_loss(regression, r, reduction='none')
  mask = d.eq(1).float()
  
  reg_loss = reg_loss * mask
  
  if class_weights is not None:
    y = rearrange(y, 'b t c -> b c t')

    high_prob = torch.amax(y, dim = 1)
    knowns = high_prob.eq(1).float()
    unknowns = high_prob.lt(1).float()
  
    reg_loss_unknowns = reg_loss * unknowns
        
    class_weights = torch.reshape(class_weights, (1, -1, 1))
    class_weights = y * class_weights
    class_weights = torch.amax(class_weights, dim = 1)
    
    reg_loss_knowns = reg_loss * knowns * class_weights
    
    reg_loss = reg_loss_unknowns + reg_loss_knowns
    
  reg_loss = torch.sum(reg_loss)
  n_pos = mask.sum()
  
  if n_pos>0:
    reg_loss = reg_loss / n_pos
    
  return reg_loss

def masked_classification_loss(class_logits, y, d, class_weights = None):
  # class_logits (Tensor): [batch, time,n_classes]
  # y (Tensor): [batch, time,n_classes]
  # d (Tensor) : [batch, time,], float tensor
  # class_weight : [n_classes,], float tensor
  
  class_logits = rearrange(class_logits, 'b t c -> b c t')
  y = rearrange(y, 'b t c -> b c t')
  
  high_prob = torch.amax(y, dim = 1)
  knowns = high_prob.eq(1).float()
  unknowns = high_prob.lt(1).float()
  
  mask = d.eq(1).float() # mask out time steps where no event is present
  
  known_class_loss = F.cross_entropy(class_logits, y, weight=class_weights, reduction='none')
  known_class_loss = known_class_loss * mask * knowns
  known_class_loss = torch.sum(known_class_loss)
  
  unknown_class_loss = F.cross_entropy(class_logits, y, weight=None, reduction='none')
  unknown_class_loss = unknown_class_loss * mask * unknowns
  unknown_class_loss = torch.sum(unknown_class_loss)
  
  class_loss = known_class_loss + unknown_class_loss
  n_pos = mask.sum()
  
  if n_pos>0:
    class_loss = class_loss / n_pos
    
  return class_loss
  
def get_class_loss_fn(args):
  dataloader_temp = get_train_dataloader(args, random_seed_shift = 0)
  class_proportions = dataloader_temp.dataset.get_class_proportions()
  class_weights = 1. / (class_proportions + 1e-6)
    
  class_weights = (1. / (np.mean(class_weights) + 1e-6)) * class_weights # normalize so average weight = 1
  
  print(f"Using class weights {class_weights}")
  
  class_weights = torch.Tensor(class_weights).to(device)
  return partial(masked_classification_loss, class_weights = class_weights)

def get_reg_loss_fn(args):
  dataloader_temp = get_train_dataloader(args, random_seed_shift = 0)
  class_proportions = dataloader_temp.dataset.get_class_proportions()
  class_weights = 1. / (class_proportions + 1e-6)
    
  class_weights = (1. / (np.mean(class_weights) + 1e-6)) * class_weights # normalize so average weight = 1
  
  class_weights = torch.Tensor(class_weights).to(device)
  return partial(masked_reg_loss, class_weights = class_weights)
                               