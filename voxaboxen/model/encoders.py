import torch
from torch import nn
import json
from einops import rearrange

def get_encoder(args):
    if not hasattr(args, "encoder_type"):
        args.encoder_type = "aves"
    if args.encoder_type == "aves":
        return AvesEmbedding(args)
    elif args.encoder_type == "hubert_base":
        return HubertBaseEmbedding(args)
    elif args.encoder_type == "frame_atst":
        return ATSTEncoder(args)
    elif args.encoder_type == "beats":
        return BEATsEncoder(args)
    else:
        raise ValueError

class EncoderBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sr=args.sr
        
    def forward(self, sig):
        pass

    def freeze(self):
        pass

    def unfreeze(self):
        pass
      
###### Wav2Vec2 family
      
class Wav2Vec2Base(EncoderBase):
    def __init__(self, args):
        super().__init__(args)
        self.sr=args.sr
        
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

class AvesEmbedding(Wav2Vec2Base):
    def __init__(self, args):
        super().__init__(args)
        from torchaudio.models import wav2vec2_model
        import torch.hub

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        config = self.load_config(args.aves_config_fp)
        self.model = wav2vec2_model(**config, aux_num_out=None)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.hub.load_state_dict_from_url(args.aves_url, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.feature_extractor.requires_grad_(False)
        self.embedding_dim = config['encoder_embed_dim']

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)

        return obj
      
class HubertBaseEmbedding(Wav2Vec2Base):
    def __init__(self, args):
        super().__init__(args)
        import torchaudio
        bundle = torchaudio.pipelines.HUBERT_BASE
        # Build the model and load pretrained weight.
        self.model = bundle.get_model()
        self.model.feature_extractor.requires_grad_(False)
        self.embedding_dim = bundle._params['encoder_embed_dim']
        
###### ATST Family
        
class ATSTEncoder(EncoderBase):
    def __init__(self, args):
        super().__init__(args)
        from voxaboxen.model.frame_atst import get_timestamp_embedding, load_model
        self.get_embedding = get_timestamp_embedding
        self.atst = load_model(args.frame_atst_weight_fp)
        self.embedding_dim = self.atst.timestamp_embedding_size
        
    def forward(self, x):
        encoding = self.get_embedding(x, self.atst)
        encoding = rearrange(encoding, 'b c t -> b t c')
        return encoding
    
    def freeze(self):
        self.atst.freeze()

    def unfreeze(self):
        self.atst.unfreeze()
        
###### BEATs Family

class BEATsEncoder(EncoderBase):
    def __init__(self, args):
        super().__init__(args)
        from voxaboxen.model.beats import BEATs, BEATsConfig
        beats_ckpt = torch.load(args.beats_checkpoint_fp, map_location='cpu')
        beats_cfg = BEATsConfig(beats_ckpt['cfg'])
        self.beats = BEATs(beats_cfg)
        self.beats.load_state_dict(beats_ckpt['model'])
        self.embedding_dim = self.beats.cfg.encoder_embed_dim
        
    def forward(self, x):
        encoding = self.beats.extract_features(x, feature_only=True)[0]
        return encoding
    
    def freeze(self):
        for name, param in self.beats.named_parameters():
            param.requires_grad = False
        self.beats.eval()

    def unfreeze(self):
        for name, param in self.beats.named_parameters():
            param.requires_grad = True
        self.beats.train()
