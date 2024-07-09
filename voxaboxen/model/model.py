import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.hub
import math
from einops import rearrange
from torchaudio.models import wav2vec2_model
import json

class AvesEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        config = self.load_config(args.aves_config_fp)
        self.model = wav2vec2_model(**config, aux_num_out=None)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.hub.load_state_dict_from_url(args.aves_url, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.feature_extractor.requires_grad_(False)
        self.sr=args.sr

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)

        return obj

    def forward(self, sig):
        # extract_feature in the torchaudio version will output all 12 layers' output, -1 to select the final one
        out = self.model.extract_features(sig)[0][-1]

        return out
      
    def freeze(self):
      for param in self.model.encoder.parameters():
          param.requires_grad = False

    def unfreeze(self):
      for param in self.model.encoder.parameters():
          param.requires_grad = True
      
class DetectionModel(nn.Module):
  def __init__(self, args, embedding_dim=768):
      super().__init__()
      self.encoder = AvesEmbedding(args)
      self.args = args
      aves_sr = args.sr // args.scale_factor
      self.detection_head = DetectionHead(args, embedding_dim = embedding_dim)
      
  def forward(self, x):
      """
      Input
        x (Tensor): (batch, time) (time at 16000 Hz, audio_sr)
      Returns
        detection_probs (Tensor): (batch, time,) (time at 50 Hz, aves_sr)
        regression (Tensor): (batch, time,) (time at 50 Hz, aves_sr)
        class_logits (Tensor): (batch, time, n_classes) (time at 50 Hz, aves_sr)

      """
      
      expected_dur_output = math.ceil(x.size(1)/self.args.scale_factor)
            
      x = x-torch.mean(x,axis=1,keepdim=True)
      feats = self.encoder(x)
      
      #aves may be off by 1 sample from expected
      pad = expected_dur_output - feats.size(1)
      if pad>0:
        feats = F.pad(feats, (0,0,0,pad), mode='reflect')
      
      detection_logits, regression, class_logits = self.detection_head(feats)
      detection_probs = torch.sigmoid(detection_logits)
      
      return detection_probs, regression, class_logits
    
  def generate_features(self, x):
      """
      Input
        x (Tensor): (batch, time) (time at 16000 Hz, audio_sr)
      Returns
        features (Tensor): (batch, time, embedding_dim) (time at 50 Hz, aves_sr)
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
  def __init__(self, args, embedding_dim=768):
      super().__init__()
      self.n_classes = len(args.label_set)
      self.head = nn.Conv1d(embedding_dim, 2+self.n_classes, args.prediction_scale_factor, stride=args.prediction_scale_factor, padding=0)
      self.args=args
      
  def forward(self, x):
      """
      Input
        x (Tensor): (batch, time, embedding_dim) (time at 50 Hz, aves_sr)
      Returns
        detection_logits (Tensor): (batch, time,) (time at 50 Hz, aves_sr)
        reg (Tensor): (batch, time,) (time at 50 Hz, aves_sr)
        class_logits (Tensor): (batch, time, n_classes) (time at 50 Hz, aves_sr)

      """
      x = rearrange(x, 'b t c -> b c t')
      x = self.head(x)
      x = rearrange(x, 'b c t -> b t c')
      detection_logits = x[:,:,0]      
      reg = x[:,:,1]
      class_logits = x[:,:,2:]
      return detection_logits, reg, class_logits
    
class DetectionModelStereo(DetectionModel):
  def __init__(self, args, embedding_dim=768):
      super().__init__(args, embedding_dim=2*embedding_dim)
      
  def forward(self, x):
    """
    Input
      x (Tensor): (batch, channels, time) (time at 16000 Hz, audio_sr)
    Returns
      detection_probs (Tensor): (batch, time,) (time at 50 Hz, aves_sr)
      regression (Tensor): (batch, time,) (time at 50 Hz, aves_sr)
      class_logits (Tensor): (batch, time, n_classes) (time at 50 Hz, aves_sr)

    """
    
    expected_dur_output = math.ceil(x.size(-1)/self.args.scale_factor)
          
    x = x-torch.mean(x,axis=-1,keepdim=True)
    feats0 = self.encoder(x[:,0,:])
    feats1 = self.encoder(x[:,1,:])
    feats = torch.cat([feats0,feats1],dim=-1)

    #aves may be off by 1 sample from expected
    pad = expected_dur_output - feats.size(1)
    if pad>0:
      feats = F.pad(feats, (0,0,0,pad), mode='reflect')
    
    detection_logits, regression, class_logits = self.detection_head(feats)
    detection_probs = torch.sigmoid(detection_logits)
    
    return detection_probs, regression, class_logits

class DetectionModelMultichannel(DetectionModel):
  # supports >1 channel, but unlike Stereo model does not assume the order of channels matters.
  def __init__(self, args, embedding_dim=768):
      super().__init__(args, embedding_dim=embedding_dim)
      self.n_classes = len(args.label_set)
      
  def forward(self, x):
    """
    Input
      x (Tensor): (batch, channels, time) (time at 16000 Hz, audio_sr)
    Returns
      detection_probs (Tensor): (batch, time,) (time at 50 Hz, aves_sr)
      regression (Tensor): (batch, time,) (time at 50 Hz, aves_sr)
      class_logits (Tensor): (batch, time, n_classes) (time at 50 Hz, aves_sr)

    """
    
#     expected_dur_output = math.ceil(x.size(-1)/self.args.scale_factor)
          
#     x = x-torch.mean(x,axis=-1,keepdim=True)
#     feats=[]
#     for i in range(x.size(1)):
#       feats.append(self.encoder(x[:,i,:]))
#     feats = sum(feats)

#     #aves may be off by 1 sample from expected
#     pad = expected_dur_output - feats.size(1)
#     if pad>0:
#       feats = F.pad(feats, (0,0,0,pad), mode='reflect')
    
#     detection_logits, regression, class_logits = self.detection_head(feats)
#     detection_probs = torch.sigmoid(detection_logits)


    expected_dur_output = math.ceil(x.size(-1)/self.args.scale_factor)
          
    x = x-torch.mean(x,axis=-1,keepdim=True)
    
    detection_logits = []
    regression = []
    class_logits = []
    
    for i in range(x.size(1)):
      feats = self.encoder(x[:,i,:])
      #aves may be off by 1 sample from expected
      pad = expected_dur_output - feats.size(1)
      if pad>0:
        feats = F.pad(feats, (0,0,0,pad), mode='reflect')

      dl, rg, cl = self.detection_head(feats)
      detection_logits.append(dl)
      regression.append(rg)
      class_logits.append(cl)
      
    detection_logits = torch.stack(detection_logits, dim=0)
    regression = torch.stack(regression, dim=0)
    class_logits = torch.stack(class_logits, dim=0)
    
    if hasattr(self.args, "segmentation_based") and self.args.segmentation_based:
      detection_logits=torch.max(detection_logits,dim=0)[0]
      regression=torch.max(regression,dim=0)[0]
      class_logits=torch.max(class_logits,dim=0)[0]
    else:
      mask_based_on_logits = torch.eq(torch.max(detection_logits, dim=0,keepdim=True)[0], detection_logits)

      detection_logits = torch.sum(detection_logits*mask_based_on_logits, dim=0)
      regression = torch.sum(regression*mask_based_on_logits,dim=0)
      class_logits = torch.sum(class_logits* mask_based_on_logits.unsqueeze(-1).repeat(1,1,1,self.n_classes), dim=0)

    detection_probs = torch.sigmoid(detection_logits)
      
    return detection_probs, regression, class_logits
  

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

    X = (X + X_aug)#[:batch_size//2,...]
    d = torch.maximum(d, d_aug)#[:batch_size//2,...]
    r = torch.maximum(r, r_aug)#[:batch_size//2,...]
    y = torch.maximum(y, y_aug)#[:batch_size//2,...]

    if args.rms_norm:
      X = X * (1/2)
    
  return X, d, r, y
      
