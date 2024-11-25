import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math
from einops import rearrange
from voxaboxen.model.encoders import get_encoder

class DetectionModel(nn.Module):
  def __init__(self, args, embedding_dim=None):
      super().__init__()
      self.is_bidirectional = args.bidirectional if hasattr(args, "bidirectional") else False
      self.is_stereo = args.stereo if hasattr(args, "stereo") else False
      self.is_segmentation = args.segmentation_based if hasattr(args, "segmentation_based") else False
      self.encoder = get_encoder(args)
      embedding_dim = self.encoder.embedding_dim
      if self.is_stereo:
          embedding_dim *= 2
      self.args = args
      aves_sr = args.sr // args.scale_factor
      self.detection_head = DetectionHead(args, embedding_dim = embedding_dim)
      if self.is_bidirectional:
          self.rev_detection_head = DetectionHead(args, embedding_dim = embedding_dim)

  def forward(self, x):
      """
      Input
        x (Tensor): (batch, time) (time at 16000 Hz, audio_sr)
      Returns
        detection_probs (Tensor): (batch, time,) (time at encoder sr)
        regression (Tensor): (batch, time,) (time at encoder sr)
        class_logits (Tensor): (batch, time, n_classes) (time at encoder sr)

      """

      expected_dur_output = math.ceil(x.size(-1)/self.args.scale_factor)

      x = x-torch.mean(x,axis=-1,keepdim=True)
      if self.is_stereo:
          feats0 = self.encoder(x[:,0,:])
          feats1 = self.encoder(x[:,1,:])
          feats = torch.cat([feats0,feats1],dim=-1)
      else:
          feats = self.encoder(x)

      #aves may be off by 1 sample from expected
      pad = expected_dur_output - feats.size(1)
      if pad>0:
        feats = F.pad(feats, (0,0,0,pad), mode='reflect')

      detection_logits, regression, class_logits = self.detection_head(feats)
      detection_probs = torch.sigmoid(detection_logits)
      if self.is_bidirectional:
          rev_detection_logits, rev_regression, rev_class_logits = self.rev_detection_head(feats)
          rev_detection_probs = torch.sigmoid(rev_detection_logits)
      else:
          rev_detection_probs = rev_regression = rev_class_logits = None
          
      return detection_probs, regression, class_logits, rev_detection_probs, rev_regression, rev_class_logits

  def generate_features(self, x):
      """
      Input
        x (Tensor): (batch, time) (time at audio sr)
      Returns
        features (Tensor): (batch, time, embedding_dim) (time at encoder sr)
      """

      expected_dur_output = math.ceil(x.size(-1)/self.args.scale_factor)

      x = x-torch.mean(x,axis=-1,keepdim=True)
      feats = self.encoder(x)

      #aves may be off by 1 sample from expected
      pad = expected_dur_output - feats.size(1)
      if pad>0:
        feats = F.pad(feats, (0,0,0,pad), mode='reflect')

      return feats

  def freeze_encoder(self):
      self.encoder.freeze()

  def unfreeze_encoder(self):
      self.encoder.unfreeze()

class DetectionHead(nn.Module):
  def __init__(self, args, embedding_dim=None):
      super().__init__()
      self.n_classes = len(args.label_set)
      self.head = nn.Conv1d(embedding_dim, 2+self.n_classes, args.prediction_scale_factor, stride=args.prediction_scale_factor, padding=0)
      self.args=args

  def forward(self, x):
      """
      Input
        x (Tensor): (batch, time, embedding_dim) (time at audio sr)
      Returns
        detection_logits (Tensor): (batch, time,) (time at encoder sr)
        reg (Tensor): (batch, time,) (time at encoder sr)
        class_logits (Tensor): (batch, time, n_classes) (time at encoder sr)

      """
      x = rearrange(x, 'b t c -> b c t')
      x = self.head(x)
      x = rearrange(x, 'b c t -> b t c')
      detection_logits = x[:,:,0]
      reg = x[:,:,1]
      class_logits = x[:,:,2:]
      return detection_logits, reg, class_logits

def rms_and_mixup(X, d, r, y, train, args):
  if args.rms_norm:
    ms = torch.mean(X ** 2, dim = -1, keepdim = True)
    ms = ms + torch.full_like(ms, 1e-6)
    rms = ms ** (-1/2)
    X = X * rms

  if args.mixup and train:
    # TODO: For mixup, add in a check that there aren't extremely overlapping vocs

    batch_size = X.size(0)

    mask = torch.full((X.size(0),1,1), 0.5, device=X.device)
    mask = torch.bernoulli(mask)

    if len(X.size()) == 2:
        X_aug = torch.flip(X, (0,)) * mask[:,:,0]
    elif  len(X.size()) == 3:
        X_aug = torch.flip(X, (0,)) * mask

    d_aug = torch.flip(d, (0,)) * mask[:,:,0]
    r_aug = torch.flip(r, (0,)) * mask[:,:,0]
    y_aug = torch.flip(y, (0,)) * mask

    X = (X + X_aug)
    d = torch.maximum(d, d_aug)
    r = torch.maximum(r, r_aug)
    y = torch.maximum(y, y_aug)
    if args.rms_norm:
      X = X * (1/2)

  return X, d, r, y

