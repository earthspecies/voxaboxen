import torch
from torch import nn
from torch.nn import functional as F
import math
from einops import rearrange
from voxaboxen.model.encoders import get_encoder

class DetectionModel(nn.Module):
    """A neural model for audio event detection and classification.

    The model consists of an transformer encoder followed by one detection
    head (two if in bidirectional mode).

    Parameters
    ----------
    args : argparse.Namespace
        Configuration arguments containing model parameters:
        - bidirectional (bool): Whether to use bidirectional processing
        - stereo (bool): Whether input is stereo audio
        - segmentation_based (bool): Whether to use segmentation-based approach
        - sr (int): Sample rate of input audio
        - scale_factor (int): Downsampling factor from audio to encoder output
        - label_set (list): List of class labels
    embedding_dim : int, optional
        Dimension of encoder embeddings, inferred from encoder if not provided
    """

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
        self.detection_head = DetectionHead(args, embedding_dim = embedding_dim)
        if self.is_bidirectional:
            self.rev_detection_head = DetectionHead(args, embedding_dim = embedding_dim)

    def forward(self, x):
        """Forward pass of the detection model.

        Parameters
        ----------
        x : torch.Tensor
            Input audio tensor of shape (batch, time) or (batch, 2, time) for stereo,
            where time is at the original audio sample rate (e.g., 16000 Hz)

        Returns
        -------
        torch.Tensor
            Probabilities that an event begins at each time step (batch, time)
        torch.Tensor
            Predicted duration of an event beginning at each time step, can
            be ignored if corresponding prediction prob is low (batch, time)
        torch.Tensor
            Predicted class logits of the event beginning at each time step, can
            be ignored if corresponding prediction prob is low
            (batch, time, n_classes)
        torch.Tensor or None
            Probabilities that an event end at each time step (batch, time),
            None if not in bidirectional mode
        torch.Tensor or None
            Predicted duration of an event ending at each time step
            can be ignored if corresponding prediction prob is low (batch, time)
            None if not in bidirectional mode
        torch.Tensor or None
            Predicted class logits of the event ending at each time step, can
            be ignored if corresponding prediction prob is low, None if not in
            bidirectional mode
        All time dimensions are at the encoder's sample rate
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

    def freeze_encoder(self):
        """Freeze encoder parameters to prevent updates during training."""
        self.encoder.freeze()

    def unfreeze_encoder(self):
        self.encoder.unfreeze()
        """Unfreeze encoder parameters to allow updates during training."""


class DetectionHead(nn.Module):
    """Detection head for making predictions from encoder outputs.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration arguments containing:
        - label_set (list): List of class labels
    embedding_dim : int
        Dimension of input embeddings
    """

    def __init__(self, args, embedding_dim=None):
        super().__init__()
        self.n_classes = len(args.label_set)
        self.head = nn.Conv1d(embedding_dim, 2+self.n_classes, 1, stride=1, padding=0)
        self.args=args

    def forward(self, x):
        """Forward pass of detection head.

        Parameters
        ----------
        x : torch.Tensor
            Input encoder features of shape (batch, time, embedding_dim)

        Returns
        -------
        torch.Tensor
            Probability score for each time step (batch, time)
        torch.Tensor
            Regression value for each time step (batch, time)
        torch.Tensor
            Real number (e.g. class logits) for each time step
            (batch, time, self.n_classes)
        """

        x = rearrange(x, 'b t c -> b c t')
        x = self.head(x)
        x = rearrange(x, 'b c t -> b t c')
        detection_logits = x[:,:,0]
        reg = x[:,:,1]
        class_logits = x[:,:,2:]
        return detection_logits, reg, class_logits

def rms_and_mixup(X, d, r, y, train, args):
    """Apply optional RMS normalization and optional mixup augmentation.

    Parameters
    ----------
    X : torch.Tensor
        Input audio features
    d : torch.Tensor
        Detection targets
    r : torch.Tensor
        Regression targets
    y : torch.Tensor
        Classification targets
    train : bool
        Whether in training mode (enables mixup)
    args : argparse.Namespace
        Configuration arguments containing:
        - rms_norm (bool): Whether to apply RMS normalization
        - mixup (bool): Whether to apply mixup augmentation

    Returns
    -------
    torch.Tensor
        Maybe rms-normalized features
    torch.Tensor
        Maybe mixuped detection targets
    torch.Tensor
        Maybe mixuped regression targets
    torch.Tensor
        Maybe mixuped classification targets
    """

    if args.rms_norm:
        ms = torch.mean(X ** 2, dim = -1, keepdim = True)
        ms = ms + torch.full_like(ms, 1e-6)
        rms = ms ** (-1/2)
        X = X * rms

    if args.mixup and train:
        # TODO: For mixup, add in a check that there aren't extremely overlapping vocs

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

