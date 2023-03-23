import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import fairseq
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
      
class DetectionModel(nn.Module):
  def __init__(self, args, embedding_dim=768):
      super().__init__()
      self.encoder = AvesEmbedding(args)
      aves_sr = args.sr // args.scale_factor
      anchor_durs_aves_samples = (aves_sr * np.array(args.anchor_durs_sec)).astype(int)
      self.detection_head = DetectionHead(anchor_durs_aves_samples, embedding_dim = embedding_dim)
      
  def forward(self, x):
      """
      Input
        x (Tensor): (batch, time) (time at 16000 Hz, audio_sr)
      Returns
        y (Tensor): (batch, time, n_anchors) (time at 50 Hz, aves_sr)
      """
      feats = self.encoder(x)
      logits = self.detection_head(feats)
      return logits
    
class DetectionHead(nn.Module):
  def __init__(self, anchor_durs_aves_samples, embedding_dim=768):
      super().__init__()
      n_anchors = len(anchor_durs_aves_samples)
      self.prediction_heads = []
      # TODO: vectorize if we want many anchor sizes
      for anchor_dur in anchor_durs_aves_samples: 
        if anchor_dur % 2 == 0:
          anchor_dur += 1
        padding = (anchor_dur - 1) // 2 # padding = 'same'
        pool = nn.AvgPool1d(anchor_dur, stride=1, padding=padding, ceil_mode=False, count_include_pad=False)
        head = nn.Conv1d(embedding_dim, 1, 1, stride=1)
        poolhead = nn.Sequential(pool, head)
        self.prediction_heads.append(poolhead)
      self.prediction_heads = nn.ModuleList(self.prediction_heads)
      
  def forward(self, x):
      """
      Input
        x (Tensor): (batch, time, embedding_dim) (time at 50 Hz, aves_sr)
      Returns
        y (Tensor): (batch, time, n_anchors) (time at 50 Hz, aves_sr)
      """
      x = rearrange(x, 'b t c -> b c t')
      outputs = []
      for prediction_head in self.prediction_heads:
        outputs.append(prediction_head(x))
      x = torch.cat(outputs, dim = 1)
      x = rearrange(x, 'b c t -> b t c')
      return x
      
      
      
      
