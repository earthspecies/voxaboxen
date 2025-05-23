"""
Classes for different encoder options
"""

import json

import torch
from einops import rearrange
from torch import nn


def get_encoder(args):
    """
    Load encoder based on args
    """
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
    elif args.encoder_type == "crnn":
        return CRNNEncoder(args)
    else:
        raise ValueError


class EncoderBase(nn.Module):
    """
    Base class for encoder used in detection model
    """

    def __init__(self, args):
        super().__init__()
        self.sr = args.sr

    def forward(self, sig):
        """
        hook for forward pass
        """
        pass

    def freeze(self):
        """
        hook for freezing encoder weights
        """
        pass

    def unfreeze(self):
        """
        hook for unfreezing encoder weights
        """
        pass


###### Wav2Vec2 family


class Wav2Vec2Base(EncoderBase):
    """
    Wav2Vec2-based encoder
    """

    def __init__(self, args):
        super().__init__(args)
        self.sr = args.sr

    def forward(self, sig):
        """
        Forward pass
        Parameters
        ----------
        sig : torch.Tensor
            audio of shape [batch, time]
        Returns
        -------
        torch.Tensor
            features of shape [batch, time, channels]
        """
        # extract_feature in the torchaudio version will output all 12 layers' output, -1 to select the final one
        out = self.model.extract_features(sig)[0][-1]

        return out

    def freeze(self):
        """
        Freeze encoder weights
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze encoder weights
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = True


class AvesEmbedding(Wav2Vec2Base):
    """
    AVES-based encoder
    """

    def __init__(self, args):
        super().__init__(args)
        import torch.hub
        from torchaudio.models import wav2vec2_model

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        config = self.load_config(args.aves_config_fp)
        self.model = wav2vec2_model(**config, aux_num_out=None)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.hub.load_state_dict_from_url(
            args.aves_url, map_location=device
        )
        self.model.load_state_dict(state_dict)
        self.model.feature_extractor.requires_grad_(False)
        self.embedding_dim = config["encoder_embed_dim"]

    def load_config(self, config_path):
        """
        Load model config json from config_path
        """
        with open(config_path, "r") as ff:
            obj = json.load(ff)

        return obj


class HubertBaseEmbedding(Wav2Vec2Base):
    """
    Hubert-based encoder
    """

    def __init__(self, args):
        super().__init__(args)
        import torchaudio

        bundle = torchaudio.pipelines.HUBERT_BASE
        # Build the model and load pretrained weight.
        self.model = bundle.get_model()
        self.model.feature_extractor.requires_grad_(False)
        self.embedding_dim = bundle._params["encoder_embed_dim"]


###### ATST Family


class ATSTEncoder(EncoderBase):
    """
    Frame-ATST based encoder
    """

    def __init__(self, args):
        super().__init__(args)
        from voxaboxen.model.frame_atst import (get_timestamp_embedding,
                                                load_model)

        self.get_embedding = get_timestamp_embedding
        self.atst = load_model(args.frame_atst_weight_fp)
        self.embedding_dim = self.atst.timestamp_embedding_size

    def forward(self, x):
        """
        Forward pass
        Parameters
        ----------
        x : torch.Tensor
            audio of shape [batch, time]
        Returns
        -------
        torch.Tensor
            features of shape [batch, time, channels]
        """
        encoding = self.get_embedding(x, self.atst)
        encoding = rearrange(encoding, "b c t -> b t c")
        return encoding

    def freeze(self):
        """
        Freeze encoder weights
        """
        self.atst.freeze()

    def unfreeze(self):
        """
        Unfreeze encoder weights
        """
        self.atst.unfreeze()


###### BEATs Family


class BEATsEncoder(EncoderBase):
    """
    BEATs-based encoder
    """

    def __init__(self, args):
        super().__init__(args)
        from voxaboxen.model.beats import BEATs, BEATsConfig

        beats_ckpt = torch.load(args.beats_checkpoint_fp, map_location="cpu")
        beats_cfg = BEATsConfig(beats_ckpt["cfg"])
        self.beats = BEATs(beats_cfg)
        self.beats.load_state_dict(beats_ckpt["model"])
        self.embedding_dim = self.beats.cfg.encoder_embed_dim

    def forward(self, x):
        """
        Forward pass
        Parameters
        ----------
        x : torch.Tensor
            audio of shape [batch, time]
        Returns
        -------
        torch.Tensor
            features of shape [batch, time, channels]
        """
        encoding = self.beats.extract_features(x, feature_only=True)[0]
        return encoding

    def freeze(self):
        """
        Freeze encoder
        """
        for name, param in self.beats.named_parameters():
            param.requires_grad = False
        self.beats.eval()

    def unfreeze(self):
        """
        Unfreeze encoder
        """
        for name, param in self.beats.named_parameters():
            param.requires_grad = True
        self.beats.train()


class CRNNEncoder(EncoderBase):
    """
    CRNN-based encoder
    """

    def __init__(self, args):
        super().__init__(args)
        from voxaboxen.model.crnn import CRNN

        self.encoder = CRNN(args)
        self.embedding_dim = self.encoder.output_dim

    def forward(self, x):
        """
        Forward pass
        Parameters
        ----------
        x : torch.Tensor
            audio of shape [batch, time]
        Returns
        -------
        torch.Tensor
            features of shape [batch, time, channels]
        """
        encoding = self.encoder(x)
        return encoding
