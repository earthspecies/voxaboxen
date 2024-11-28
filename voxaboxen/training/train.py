import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm
from functools import partial
import os
from einops import rearrange
import yaml

from voxaboxen.evaluation.plotters import plot_eval
from voxaboxen.evaluation.evaluation import predict_and_generate_manifest, evaluate_based_on_manifest
from voxaboxen.data.data import get_train_dataloader, get_val_dataloader
from voxaboxen.model.model import rms_and_mixup

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
  import warnings
  warnings.warn("Only using CPU! Check CUDA")

def train(model, args):

  detection_loss_fn = get_detection_loss_fn(args)
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

      train_evals_by_epoch = {i : e for i, e in enumerate(train_evals)}
      train_evals_fp = os.path.join(args.experiment_dir, "train_history.yaml")
      with open(train_evals_fp, 'w') as f:
        yaml.dump(train_evals_by_epoch, f)

      if use_val:
        eval_scores = val_epoch(model, t, val_dataloader, args)
        # TODO: maybe plot evals for other pred_types
        val_evals.append(eval_scores['fwd'].copy())
        plot_eval(train_evals, learning_rates, args, val_evals=val_evals)

        val_evals_by_epoch = {i : e for i, e in enumerate(val_evals)}
        val_evals_fp = os.path.join(args.experiment_dir, "val_history.yaml")
        with open(val_evals_fp, 'w') as f:
          yaml.dump(val_evals_by_epoch, f)
      else:
        plot_eval(train_evals, learning_rates, args)
      scheduler.step()

      if use_val and args.early_stopping:
        current_f1 = eval_scores['comb']['f1'] if model.is_bidirectional else eval_scores['fwd']['f1']
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
              os.path.join(args.experiment_dir, "model.pt"),
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
              os.path.join(args.experiment_dir, "model.pt"),
          )


  print("Done!")

  cp = torch.load(os.path.join(args.experiment_dir, "model.pt"))
  model.load_state_dict(cp["model_state_dict"])

  # resave validation with best model
  if use_val:
    val_epoch(model, args.n_epochs, val_dataloader, args)

  return model

def lf(dets, det_preds, regs, reg_preds, y, y_preds, args, det_loss_fn, reg_loss_fn, class_loss_fn):
    # We mask out loss from each end of the clip, so the model isn't forced to learn to detect events that are partially cut off.
    # This does not affect inference, because during inference we overlap clips at 50%
    
    end_mask_perc = args.end_mask_perc
    end_mask_dur = int(det_preds.size(1)*end_mask_perc)

    det_preds_clipped = det_preds[:,end_mask_dur:-end_mask_dur]
    dets_clipped = dets[:,end_mask_dur:-end_mask_dur]

    reg_preds_clipped = reg_preds[:,end_mask_dur:-end_mask_dur]
    regs_clipped = regs[:,end_mask_dur:-end_mask_dur]

    y_clipped = y[:,end_mask_dur:-end_mask_dur,:]

    detection_loss = det_loss_fn(det_preds_clipped, dets_clipped, pos_loss_weight=args.pos_loss_weight)
    reg_loss = reg_loss_fn(reg_preds_clipped, regs_clipped, dets_clipped, y_clipped)
    if len(args.label_set)==1:
        class_loss = torch.tensor(0)
    else:
        y_preds_clipped = y_preds[:,end_mask_dur:-end_mask_dur,:]
        class_loss = class_loss_fn(y_preds_clipped, y_clipped, dets_clipped)
    return detection_loss, reg_loss, class_loss

def train_epoch(model, t, dataloader, detection_loss_fn, reg_loss_fn, class_loss_fn, optimizer, args):
    model.train()
    if t < args.unfreeze_encoder_epoch:
      model.freeze_encoder()
    else:
      model.unfreeze_encoder()


    evals = {}
    train_loss = 0; losses = []; detection_losses = []; regression_losses = []; class_losses = []
    rev_train_loss = 0; rev_losses = []; rev_detection_losses = []; rev_regression_losses = []; rev_class_losses = []

    data_iterator = tqdm.tqdm(dataloader)
    #for i, (X, d, r, y, rev_d, rev_r, rev_y) in enumerate(data_iterator):
    for i, batch in enumerate(data_iterator):
      num_batches_seen = i
      batch = [item.to(device, dtype=torch.float) for item in batch]
      X, d, r, y = batch[:4]      
      X, d, r, y = rms_and_mixup(X, d, r, y, True, args)
      probs, regression, class_logits, rev_probs, rev_regression, rev_class_logits = model(X)
      #model_outputs = model(X)
      #probs, regression, class_logits = model_outputs[:3]
      detection_loss, reg_loss, class_loss = lf(d, probs, r, regression, y, class_logits, args=args, det_loss_fn=detection_loss_fn, reg_loss_fn=reg_loss_fn, class_loss_fn=class_loss_fn)

      loss = args.rho * class_loss + detection_loss + args.lamb * reg_loss
      train_loss += loss.item()
      losses.append(loss.item())
      detection_losses.append(detection_loss.item())
      regression_losses.append(args.lamb * reg_loss.item())
      class_losses.append(args.rho * class_loss.item())

      pbar_str = f"loss {np.mean(losses[-10:]):.5f}, det {np.mean(detection_losses[-10:]):.5f}, reg {np.mean(regression_losses[-10:]):.5f}, class {np.mean(class_losses[-10:]):.5f}"

      if model.is_bidirectional:
          assert all(x is not None for x in [rev_probs, rev_regression, rev_class_logits])
          rev_d, rev_r, rev_y = batch[4:]
          #rev_probs, rev_regression, rev_class_logits = model_outputs[3:]
          _, rev_d, rev_r, rev_y = rms_and_mixup(X, rev_d, rev_r, rev_y, True, args)

          rev_detection_loss, rev_reg_loss, rev_class_loss = lf(rev_d, rev_probs, rev_r, rev_regression, rev_y, rev_class_logits, args=args, det_loss_fn=detection_loss_fn, reg_loss_fn=reg_loss_fn, class_loss_fn=class_loss_fn)
          rev_loss = args.rho * rev_class_loss + rev_detection_loss + args.lamb * rev_reg_loss
          rev_train_loss += rev_loss.item()
          rev_losses.append(rev_loss.item())
          rev_detection_losses.append(rev_detection_loss.item())
          rev_regression_losses.append(args.lamb * rev_reg_loss.item())
          rev_class_losses.append(args.rho * rev_class_loss.item())
          loss = (loss + rev_loss)/2

          pbar_str += f" revloss {np.mean(rev_losses[-10:]):.5f}, revdet {np.mean(rev_detection_losses[-10:]):.5f}, revreg {np.mean(rev_regression_losses[-10:]):.5f}, revclass {np.mean(rev_class_losses[-10:]):.5f}"
      else:
          assert all(x is None for x in [rev_probs, rev_regression, rev_class_logits])


      optimizer.zero_grad()
      loss.backward()

      optimizer.step()
      if i > 10:
        data_iterator.set_description(pbar_str)

      if args.is_test and i == 15: break

    train_loss = train_loss / num_batches_seen
    evals['loss'] = float(train_loss)

    print(f"Epoch {t} | Train loss: {train_loss:1.3f}")
    return model, evals

def val_epoch(model, t, dataloader, args):
    model.eval()

    manifest = predict_and_generate_manifest(model, dataloader, args, verbose = False)
    e, _ = evaluate_based_on_manifest(manifest, args, output_dir=os.path.join(args.experiment_dir, 'val_results'), iou=args.model_selection_iou, class_threshold=args.model_selection_class_threshold, comb_discard_threshold=args.comb_discard_thresh)

    print(f"Epoch {t} | val@{args.model_selection_iou}IoU:")
    evals = {}
    for pt in e.keys():
        evals[pt] = {k:[] for k in ['precision','recall','f1', 'precision_seg', 'recall_seg', 'f1_seg']}
        for k in ['precision','recall','f1', 'precision_seg', 'recall_seg', 'f1_seg']:
          for l in args.label_set:
            m = e[pt]['summary'][l][k]
            evals[pt][k].append(m)
          evals[pt][k] = float(np.mean(evals[pt][k]))

        print(f"{pt}prec: {evals[pt]['precision']:1.3f} {pt}rec: {evals[pt]['recall']:1.3f} {pt}F1: {evals[pt]['f1']:1.3f} {pt}prec_seg: {evals[pt]['precision_seg']:1.3f} {pt}rec_seg: {evals[pt]['recall_seg']:1.3f} {pt}F1_seg: {evals[pt]['f1_seg']:1.3f}", end=' ')
    print()
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

def segmentation_loss(class_logits, y, d, class_weights=None):
  # class_logits (Tensor): [batch, time,n_classes]
  # y (Tensor): [batch, time,n_classes]
  # d (Tensor) : [batch, time,], float tensor
  # class_weight : [n_classes,], float tensor
  
  default_focal_loss = torchvision.ops.sigmoid_focal_loss(class_logits, y, reduction='mean')
  return default_focal_loss

def get_class_loss_fn(args):
  if hasattr(args,"segmentation_based") and args.segmentation_based:
    return segmentation_loss
  else:
    dataloader_temp = get_train_dataloader(args, random_seed_shift = 0)
    class_proportions = dataloader_temp.dataset.get_class_proportions()
    class_weights = 1. / (class_proportions + 1e-6)
    class_weights = class_weights * (class_proportions>0) # ignore weights for unrepresented classes
    
    class_weights = (1. / (np.mean(class_weights) + 1e-6)) * class_weights # normalize so average weight = 1
    print(f"Using class weights {class_weights}") 

    class_weights = torch.Tensor(class_weights).to(device)
    return partial(masked_classification_loss, class_weights = class_weights)

def get_reg_loss_fn(args):
  if hasattr(args,"segmentation_based") and args.segmentation_based:
    def zrl(regression, r, d, y, class_weights = None):
      return torch.tensor(0.)
    return zrl
  else:
    dataloader_temp = get_train_dataloader(args, random_seed_shift = 0)
    class_proportions = dataloader_temp.dataset.get_class_proportions()
    class_weights = 1. / (class_proportions + 1e-6)
    class_weights = class_weights * (class_proportions>0) # ignore weights for unrepresented classes
    
    class_weights = (1. / (np.mean(class_weights) + 1e-6)) * class_weights # normalize so average weight = 1

    class_weights = torch.Tensor(class_weights).to(device)
    return partial(masked_reg_loss, class_weights = class_weights)
                               
def get_detection_loss_fn(args):
  if hasattr(args,"segmentation_based") and args.segmentation_based:
    def zdl(pred, gt, pos_loss_weight = 1):
      return torch.tensor(0.)
    return zdl
  else:
    return modified_focal_loss
