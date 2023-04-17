import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import fairseq
import math
from einops import rearrange

class AvesEmbedding(nn.Module):
    """ Uses AVES Hubert to embed sounds """

    def __init__(self, args):
        super().__init__()
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.model_weight_fp])
        self.model = models[0]
        self.sr=args.sr

    def forward(self, x):
        """
        Input
          x (Tensor): (batch, time) (time at 16000 Hz, audio_sr)
        Returns
          feats (Tensor): (batch, time, embedding_dim) (time at 50 Hz, aves_sr)
        """
        feats = self.model.extract_features(x)[0]
        return feats
      
    def freeze(self):
      for param in self.model.parameters():
          param.requires_grad = False
          
    def unfreeze(self):
      for param in self.model.parameters():
          param.requires_grad = True
      
class DetectionModel(nn.Module):
  def __init__(self, args, embedding_dim=768):
      super().__init__()
      self.encoder = AvesEmbedding(args)
      self.args = args
      aves_sr = args.sr // args.scale_factor
      prediction_scale_factor = args.prediction_scale_factor
      self.detection_head = DetectionHead(args, embedding_dim = embedding_dim)
      
  def forward(self, x):
      """
      Input
        x (Tensor): (batch, time) (time at 16000 Hz, audio_sr)
      Returns
        y (Tensor): (batch, time, n_anchors) (time at 50 Hz, aves_sr)
      """
      
      expected_dur_output = math.ceil(x.size(1)/self.args.scale_factor)
            
      x = x-torch.mean(x,axis=1,keepdim=True)
      feats = self.encoder(x)
      
      #aves may be off by 1 sample from expected
      pad = expected_dur_output - feats.size(1)
      if pad>0:
        feats = F.pad(feats, (0,0,0,pad), mode='reflect')
      
      logits, regression = self.detection_head(feats)
      return logits, regression
    
  def freeze_encoder(self):
      self.encoder.freeze()
          
  def unfreeze_encoder(self):
      self.encoder.unfreeze()

class DetectionHead(nn.Module):
  def __init__(self, args, embedding_dim=768):
      super().__init__()
      self.head = nn.Conv1d(embedding_dim, 3, args.prediction_scale_factor, stride=args.prediction_scale_factor, padding=0)
      self.args=args
      
  def forward(self, x):
      """
      Input
        x (Tensor): (batch, time, embedding_dim) (time at 50 Hz, aves_sr)
      Returns
        logits (Tensor): (batch, time) (time at 50 Hz, aves_sr)
        reg (Tensor): (batch, time, 2) (time at 50 Hz, aves_sr)
      """
      x = rearrange(x, 'b t c -> b c t')
      x = self.head(x)
      x = rearrange(x, 'b c t -> b t c')
      logits = x[:,:,0]      
      reg = x[:,:,1:]
      return logits, reg
      
      

def preprocess_and_augment(X, y, r, train, args):
  if args.rms_norm:
    rms = torch.mean(X ** 2, dim = 1, keepdim = True) ** (-1/2)
    X = X * rms
    
  if args.mixup and train:
    X_aug = torch.flip(X, (0,))
    r_aug = torch.flip(r, (0,))
    y_aug = torch.flip(y, (0,))
    
    X = (X + X_aug) / 2
    r = torch.maximum(r, r_aug)
    y = torch.maximum(y, y_aug)
    
  return X
      
