"""
Based on https://github.com/Audio-WestlakeU/audiossl/tree/6d7d00ae92dff9d232c492213d49c32df341f294/audiossl/methods/atstframe
"""

import argparse
import math
import warnings
from functools import partial
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torchaudio
from einops.layers.torch import Rearrange
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import transforms

# from transformers.optimization import AdamW

N_BLOCKS = 12


class CustomAudioTransform:
    """
    See original frame-atst package for documentation
    """

    def __repr__(self) -> str:
        """
        Returns
        ------
        str
        """
        return self.__class__.__name__ + "()"


# class Identity(CustomAudioTransform):
#     """
#     See original frame-atst package for documentation
#     """

#     def __call__(self, signal):
#         return signal


# class GaussianNoise(CustomAudioTransform):
#     """
#     See original frame-atst package for documentation
#     """

#     def __init__(self, g) -> None:
#         self.g = g

#     def __call__(self, signal):
#         """
#         See original frame-atst package for documentation
#         """
#         return signal + self.g * torch.randn_like(signal)


# class PadToSize(CustomAudioTransform):
#     """
#     See original frame-atst package for documentation
#     """

#     def __init__(self, size: int):
#         self.size = size

#     def __call__(self, signal):
#         """
#         See original frame-atst package for documentation
#         """
#         if signal.shape[1] < self.size:
#             signal = F.pad(signal, (0, self.size - signal.shape[1]))
#         return signal


# class ToSizeN(CustomAudioTransform):
#     """
#     See original frame-atst package for documentation
#     """

#     def __init__(self, size: int):
#         self.size = size

#     def __call__(self, signal):
#         """
#         See original frame-atst package for documentation
#         """
#         n = signal.shape[1] // self.size
#         m = signal.shape[1] % self.size
#         if m > self.size // 2 or n == 0:
#             signal = F.pad(signal, (0, self.size * (n + 1) - signal.shape[1]))
#         else:
#             signal = F.pad(signal, (0, self.size * n - signal.shape[1]))
#         return signal


# class CentralCrop(CustomAudioTransform):
#     """
#     See original frame-atst package for documentation
#     """

#     def __init__(self, size: int, pad: bool = True):
#         self.size = size
#         self.pad = pad

#     def __call__(self, signal):
#         """
#         See original frame-atst package for documentation
#         """
#         if signal.shape[-1] < self.size:
#             if self.pad:
#                 signal = F.pad(signal, (0, self.size - signal.shape[-1]))
#             return signal

#         start = (signal.shape[-1] - self.size) // 2
#         if len(signal.shape) > 1:
#             return signal[:, start : start + self.size]
#         else:
#             return signal[start : start + self.size]


# class RandomCrop(CustomAudioTransform):
#     """
#     See original frame-atst package for documentation
#     """

#     def __init__(self, size: int, pad: bool = True):
#         self.size = size
#         self.pad = pad

#     def __call__(self, signal):
#         """
#         See original frame-atst package for documentation
#         """
#         if signal.shape[1] < self.size:
#             if self.pad:
#                 signal = F.pad(signal, (0, self.size - signal.shape[-1]))
#             return signal
#         start = np.random.randint(0, signal.shape[-1] - self.size + 1)
#         return signal[:, start : start + self.size]


# class Normalize(CustomAudioTransform):
#     """
#     See original frame-atst package for documentation
#     """

#     def __init__(self, std_mean=None, reduce_dim=None):
#         self.std_mean = std_mean
#         self.reduce_dim = reduce_dim

#     def __call__(self, input):
#         """
#         assuming input has shape [batch,nmels,time]
#         """
#         std, mean = None, None
#         if self.std_mean is None:
#             if self.reduce_dim is not None:
#                 std, mean = torch.std_mean(input, dim=self.reduce_dim, keepdim=True)
#             else:
#                 std, mean = torch.std_mean(input)
#         else:
#             std, mean = self.std_mean
#         output = input - mean
#         output = output / (std + 1e-6)
#         return output


class MinMax(CustomAudioTransform):
    """
    See original frame-atst package for documentation
    """

    def __init__(self, min: float, max: float) -> None:
        self.min = min
        self.max = max

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        See original frame-atst package for documentation

        Returns
        ------
        torch.Tensor
        """
        min_, max_ = None, None
        if self.min is None:
            min_ = torch.min(input)
            max_ = torch.max(input)
        else:
            min_ = self.min
            max_ = self.max
        input = (input - min_) / (max_ - min_) * 2.0 - 1.0
        return input


# class div(CustomAudioTransform):
#     """
#     See original frame-atst package for documentation
#     """

#     def __init__(self, value=100):
#         self.value = value

#     def __call__(self, input):
#         """
#         See original frame-atst package for documentation
#         """
#         input /= 100
#         return input


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """
    See original frame-atst package for documentation

    Returns
    ------
    torch.Tensor
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        See original frame-atst package for documentation

        Returns
        --------
        torch.Tensor
        """
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    See original frame-atst package for documentation
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        See original frame-atst package for documentation

        Returns
        -------
        Tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    See original frame-atst package for documentation
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        See original frame-atst package for documentation

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_tokens, embed_dim).
        mask : Optional[torch.Tensor]
            Optional attention mask

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - Output tensor after multi-head attention and projection
            - Attention weights
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn += mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """See original frame-atst package for documentation"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self, x: Tensor, length: Optional[Tensor] = None, return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        See original frame-atst package for documentation

        Parameters
        ----------
        x : torch.Tensor
        length : Optional[torch.Tensor], optional
            Optional tensor of sequence lengths (used to construct attention masks).
        return_attention : bool, optional
            Whether to return attention weights in addition to the output tensor.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        """
        if length is not None:
            mask_att = get_attention_mask(x, length)
        else:
            mask_att = None

        y, attn = self.attn(self.norm1(x), mask_att)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x


def get_attention_mask(x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """
    See original frame-atst package for documentation

    Returns
    -------
    torch.Tensor
        Attention mask tensor of shape (batch_size, 1, seq_len, seq_len),
        with large negative values in padded positions.
    """
    batch_size, max_len, _ = x.shape
    # create mask for padded elements and zero-out them
    mask = (
        torch.arange(max_len, device=length.device).expand(batch_size, max_len)
        >= length[:, None]
    )
    # extend the mask to attention shape and set weight
    mask = -10000.0 * mask[:, None, None, :]
    mask = mask.expand(batch_size, 1, max_len, max_len).to(x.device)
    return mask


def _no_grad_trunc_normal_(
    tensor: torch.Tensor, mean: float, std: float, a: float, b: float
) -> torch.Tensor:
    """
    See original frame-atst package for documentation

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to fill with truncated normal values.
    mean : float
        Mean of the normal distribution.
    std : float
        Standard deviation of the normal distribution.
    a : float
        Minimum cutoff value.
    b : float
        Maximum cutoff value.

    Returns
    -------
    torch.Tensor
        The input tensor filled with truncated normal values.
    """

    # Cut & paste from PyTorch official master
    # until it's in a few official releases - RW
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x: float) -> float:
        """Computes standard normal cumulative distribution function

        Returns
        -------
        float
        """
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        ll = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * ll - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    """
    Fills the input tensor with values drawn from a truncated normal distribution.

    Returns
    -------
    Tensor
        The tensor filled with values from the truncated normal distribution.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_num_patches(
    height: int = 64, width: int = 1001, patch_height: int = 16, patch_width: int = 16
) -> int:
    """
    Computes the number of non-overlapping patches that fit into a 2D input.

    Returns
    -------
    int
    """
    return (height // patch_height) * (width // patch_width)


class PatchEmbed(nn.Module):
    """See original frame-atst package for documentation"""

    def __init__(
        self,
        patch_height: int = 64,
        patch_width: int = 4,
        embed_dim: int = 768,
        input_dim: int = 1,
    ) -> None:
        """
        Initializes the patch embedding layer using a 2D convolution.

        Parameters
        ----------
        patch_height : int, optional
            Height of each patch. Default is 64.
        patch_width : int, optional
            Width of each patch. Default is 4.
        embed_dim : int, optional
            Dimension of the output embeddings. Default is 768.
        input_dim : int, optional
            Number of input channels (e.g., 1 for grayscale spectrograms). Default is 1.
        """
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.proj = nn.Conv2d(
            input_dim,
            embed_dim,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
        )

    def forward(
        self, melspec: torch.Tensor, length: Optional[torch.Tensor] = None
    ) -> Tuple[None, Tensor, Optional[Tensor]]:
        """
        See original frame-atst package for documentation

        Parameters
        ----------
        melspec : torch.Tensor
        length : Optional[torch.Tensor], optional

        Returns
        -------
        Tuple[None, torch.Tensor, Optional[torch.Tensor]]
            - None (placeholder for compatibility)
            - Patch embeddings of shape (batch_size, num_patches, embed_dim)
            - Patch lengths tensor if `length` is provided, otherwise None
        """
        height = melspec.shape[2] - melspec.shape[2] % self.patch_height
        # width = melspec.shape[3] - melspec.shape[3] % self.patch_width
        patch_embed = self.proj(melspec).squeeze(2).permute(0, 2, 1)

        if length is not None:
            patch_length = (height // self.patch_height) * (
                (length - length % self.patch_width) // self.patch_width
            )
        else:
            patch_length = None

        return None, patch_embed, patch_length


class PatchEmbed_v2(nn.Module):
    """See original frame-atst package for documentation"""

    def __init__(
        self,
        patch_height: int = 64,
        patch_width: int = 4,
        embed_dim: int = 768,
        input_dim: int = 1,
    ) -> None:
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_maker = Rearrange(
            "b c (h p1) (w p2) -> b (w h) (p1 p2 c)", p1=patch_height, p2=patch_width
        )
        self.patch_embed = nn.Linear(patch_height * patch_width * input_dim, embed_dim)

    def forward(
        self, melspec: torch.Tensor, length: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        See original frame-atst package for documentation

        Parameters
        ----------
        melspec : torch.Tensor
        length : Optional[torch.Tensor], optional

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            - patch : Tensor of shape (batch_size, num_patches, flattened_patch_dim)
            - patch_embed : Tensor of shape (batch_size, num_patches, embed_dim)
            - patch_length : Optional tensor of shape (batch_size,), or None
        """
        height = melspec.shape[2] - melspec.shape[2] % self.patch_height
        width = melspec.shape[3] - melspec.shape[3] % self.patch_width
        patch = self.patch_maker(melspec[:, :, :height, :width])
        patch_embed = self.patch_embed(patch)

        if length is not None:
            patch_length = (height // self.patch_height) * (
                (length - length % self.patch_width) // self.patch_width
            )
        else:
            patch_length = None

        return patch, patch_embed, patch_length


class FrameAST(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        nprompt: int = 0,
        spec_h: int = 64,
        spec_w: int = 1001,
        patch_w: int = 16,
        patch_h: int = 16,
        pos_type: str = "cut",
        avg_blocks: int = 0,
        in_chans: int = 1,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        patch_embed: str = "Linear",
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.spec_w = spec_w
        self.spec_h = spec_h
        self.embed_dim = embed_dim
        self.patch_w = patch_w
        self.patch_h = patch_h

        self.pos_type = pos_type
        self.avg_blocks = avg_blocks

        if patch_embed == "Linear":
            self.patch_embed = PatchEmbed_v2(patch_h, patch_w, embed_dim)
        elif patch_embed == "CNN":
            self.patch_embed = PatchEmbed(patch_h, patch_w, embed_dim)
        else:
            raise NotImplementedError(
                "patch_embed={} not implemented".format(patch_embed)
            )

        self.mask_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # hack
        self.nprompt = nprompt
        if self.nprompt > 0:
            self.prompt_embed = nn.Parameter(
                torch.zeros(1, self.nprompt, self.embed_dim)
            )
            trunc_normal_(self.prompt_embed, std=0.02)

        num_patches = get_num_patches(spec_h, spec_w, patch_h, patch_w)
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm_frame = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.mask_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """See original frame-atst package for documentation"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(
        self,
        x: Tensor,
        mask_index: Optional[Tensor],
        length: Optional[Tensor],
        mask: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, int, int, Optional[Tensor]]:
        """
        See original frame-atst package for documentation

        Parameters
        ----------
        x : torch.Tensor
        mask_index : Optional[torch.Tensor]
        length : Optional[torch.Tensor]
        mask : bool, optional

        Returns
        -------
        Tuple[Tensor,Tensor, Tensor, int, int, Optional[Tensor]]
            - Token sequence with positional encoding applied, shape (B, T, C)
            - Positional encoding tensor, shape (B, T, C)
            - Raw patch tokens before embedding
            - Original height of spectrogram
            - Original width of spectrogram
            - Patch lengths, or None
        """
        B, nc, h, w = x.shape
        mel_patches, x, patch_length = self.patch_embed(
            x, length
        )  # patch linear embedding
        B, T, C = x.shape

        if (mask_index is not None) and mask:
            mask_index_expand = (
                mask_index.unsqueeze(2).expand(B, T, self.embed_dim).float()
            )
            x = (
                1 - mask_index_expand
            ) * x + mask_index_expand * self.mask_embed.expand(B, T, C)

        # add positional encoding to each token
        if self.pos_type == "cut":
            pos = self.pos_embed[:, 1 : T + 1, :].expand(B, -1, -1)
            x = x + pos
        else:
            pos = self.interpolate_pos_encoding(x, h, w)
            x = x + pos[:, 1:]

        # pos = self.pos_embed[:,1:T+1,:].expand(B,-1,-1)
        # x = x + pos

        return self.pos_drop(x), pos, mel_patches, h, w, patch_length

    def freeze(self) -> None:
        """See original frame-atst package for documentation"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """See original frame-atst package for documentation"""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        x: Tensor,
        mask_index: Optional[Tensor] = None,
        mask_input: bool = True,
        length: Optional[Tensor] = None,
    ) -> Tensor:
        """
        See original frame-atst package for documentation

        Parameters
        ----------
        x : torch.Tensor
        mask_index : Optional[torch.Tensor], optional
        mask_input : bool, optional
        length : Optional[torch.Tensor], optional

        Returns
        -------
        torch.Tensor
            The masked frame-level representations after transformer processing,
            shape depends on the number of masked positions.
        """
        x, pos, mel_patches, h, w, patch_length = self.prepare_tokens(
            x, mask_index, length, mask_input
        )

        length_mask = torch.arange(x.shape[1]).to(x.device) < patch_length.unsqueeze(1)
        length_mask = length_mask.to(x.device)
        mask_index = mask_index & length_mask

        if self.nprompt > 0:
            x = torch.cat([self.prompt_embed.expand(x.shape[0], -1, -1), x], dim=1)

        avg_x = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, patch_length + self.nprompt)
            if self.avg_blocks > 0:
                if i >= len(self.blocks) - self.avg_blocks:
                    avg_x.append(F.instance_norm(x.transpose(1, 2)).transpose(1, 2))

        if self.avg_blocks > 0:
            avg_x = torch.mean(torch.stack(avg_x), dim=0)
            frame_repr = avg_x
        else:
            frame_repr = self.norm_frame(x)

        return frame_repr[:, self.nprompt :][mask_index]

    def get_cls(self, x: Tensor, length: Optional[Tensor] = None) -> Tensor:
        """
        See original frame-atst package for documentation

        Parameters
        ----------
        x : torch.Tensor
        length : Optional[torch.Tensor], optional

        Returns
        -------
        torch.Tensor
        """
        x, pos, mel_patches, h, w, patch_length = self.prepare_tokens(
            x, None, length, False
        )

        if self.nprompt > 0:
            x = torch.cat([self.prompt_embed.expand(x.shape[0], -1, -1), x], dim=1)

        for _, blk in enumerate(self.blocks):
            x = blk(x, patch_length + self.nprompt)

        frame_repr = self.norm_frame(x)

        return torch.mean(frame_repr[:, : self.nprompt], dim=1)

    def interpolate_pos_encoding(self, x: Tensor, h: int, w: int) -> Tensor:
        """
        See original frame-atst package for documentation

        Parameters
        ----------
        x : torch.Tensor
        h : int
        w : int

        Returns
        -------
        torch.Tensor
            Positional encoding tensor interpolated to match input shape.
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == self.spec_w and h == self.spec_h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_width
        h0 = h // self.patch_embed.patch_height
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, self.spec_h // self.patch_h, self.spec_w // self.patch_w, dim
            ).permute(0, 3, 1, 2),
            scale_factor=(
                h0 / (self.spec_h // self.patch_h),
                w0 / (self.spec_w // self.patch_w),
            ),
            mode="bicubic",
        )
        assert (
            int(h0) == patch_pos_embed.shape[-2]
            and int(w0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def get_last_selfattention(self, x: Tensor) -> List[Tensor]:
        """
        See original frame-atst package for documentation

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        List[torch.Tensor]
            A list of attention maps (one per transformer block),
        """
        x, _, _, _, _, _ = self.prepare_tokens(
            x, mask_index=None, length=None, mask=False
        )
        atts = []
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x, att = blk(x, return_attention=True)
                atts.append(att)
            else:
                x, att = blk(x, return_attention=True)
                atts.append(att)
                return atts
                # return attention of the last block

    def get_intermediate_layers(
        self, x: Tensor, length: Optional[Tensor], n: int = 1, scene: bool = True
    ) -> Tensor:
        """
        See original frame-atst package for documentation

        Parameters
        ----------
        x : torch.Tensor
        length : Optional[torch.Tensor]
        n : int, optional
        scene : bool, optional

        Returns
        -------
        torch.Tensor
        """
        x, _, _, _, _, patch_length = self.prepare_tokens(
            x, mask_index=None, length=length, mask=False
        )
        # we return the output tokens from the `n` last blocks
        output = []
        if self.nprompt > 0:
            x = torch.cat([self.prompt_embed.expand(x.shape[0], -1, -1), x], dim=1)
        for i, blk in enumerate(self.blocks):
            x = blk(x, patch_length + self.nprompt)

            if len(self.blocks) - i <= n:
                norm_x = self.norm_frame(x)
                if scene:
                    length_mask = torch.arange(x.shape[1] - self.nprompt).to(
                        x.device
                    ) < patch_length.unsqueeze(1)
                    avg = torch.sum(
                        norm_x[:, self.nprompt :] * length_mask.unsqueeze(-1), dim=1
                    ) / (patch_length.unsqueeze(-1) + 1e-6)
                    # negative = (~length_mask) * -1e10
                    # max = torch.max(norm_x[:,self.nprompt:]
                    #   +negative.unsqueeze(-1),1).values
                    output.append(avg)
                    if self.nprompt > 0:
                        output.append(torch.mean(x[:, : self.nprompt], dim=1))
                else:
                    output.append(norm_x[:, self.nprompt :])

        return torch.cat(output, dim=-1)


def build_mlp(
    num_layers: int, input_dim: int, mlp_dim: int, output_dim: int, last_bn: bool = True
) -> nn.Sequential:
    """
    Build a multi-layer perceptron (MLP) with optional final BatchNorm.

    Parameters
    ----------
    num_layers : int
    input_dim : int
    mlp_dim : int
    output_dim : int
    last_bn : bool, optional

    Returns
    -------
    nn.Sequential
    """
    mlp = []
    for ll in range(num_layers):
        dim1 = input_dim if ll == 0 else mlp_dim
        dim2 = output_dim if ll == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if ll < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)


def byol_loss_func(p: torch.Tensor, z: torch.Tensor, simplified: bool = True) -> Tensor:
    """
    Computes BYOL loss
    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        simplified (bool): faster computation, but with same result. Defaults to True.
    Returns:
        torch.Tensor: BYOL loss.
    """

    if simplified:
        return 2 - 2 * F.cosine_similarity(p, z, dim=-1).mean()

    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)

    return 2 - 2 * (p * z).sum(dim=1).mean()


def compute_var(y: Tensor) -> Tensor:
    """See original frame-atst package for documentation

    Returns
    ----------
    Tensor
    """
    y = y.view(-1, y.size(-1))
    zc = torch.tensor(y.size(0)).cuda()
    zs = y.sum(dim=0)
    zss = (y**2).sum(dim=0)

    torch.distributed.all_reduce(zc)
    torch.distributed.all_reduce(zs)
    torch.distributed.all_reduce(zss)

    var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
    return torch.sqrt(var + 1e-6)


class ByolLoss(nn.Module):
    """See original frame-atst package for documentation"""

    def __init__(self, symmetric: bool) -> None:
        super().__init__()
        self.symmetric = symmetric

    def forward(
        self, student: nn.Module, teacher: nn.Module
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Computes the BYOL-style loss between student and teacher model outputs,
        along with standard deviation metrics for the normalized outputs.

        Parameters
        ----------
        student : nn.Module
            The student network output tensor.
        teacher : nn.Module
            The teacher network output tensor.

        Returns
        -------
        tuple of Tensor
            A tuple containing:
            - total_loss_frm : Tensor
                The computed BYOL loss between student and teacher.
            - std_frm_stu : Tensor
                The mean standard deviation of the normalized student output.
            - std_frm_tea : Tensor
                The mean standard deviation of the normalized teacher output.
        """
        stu_frm = student
        tea_frm = teacher

        std_frm_stu = compute_var(F.normalize(stu_frm, dim=-1)).mean()
        std_frm_tea = compute_var(F.normalize(tea_frm, dim=-1)).mean()

        if self.symmetric:
            stu_frm = stu_frm.chunk(2)
            tea_frm = tea_frm.chunk(2)
            total_loss_frm = 0
            n_loss_terms = 0
            for iq, q in enumerate(tea_frm):
                for iv, v in enumerate(stu_frm):
                    if iq == iv:
                        continue
                    loss = byol_loss_func(q, v, simplified=False)
                    n_loss_terms += 1
                    total_loss_frm += loss
            total_loss_frm /= n_loss_terms

        else:
            total_loss_frm = byol_loss_func(tea_frm, stu_frm)
        return total_loss_frm, std_frm_stu, std_frm_tea


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        projector: str = "mlp",
        predictor: bool = True,
    ) -> None:
        """
        Initializes the MultiCropWrapper
        """
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification

        self.encoder = encoder
        if projector == "mlp":
            self.projector = build_mlp(2, embed_dim, 4096, 256, last_bn=False)
        elif projector == "linear":
            self.projector = nn.Linear(embed_dim, embed_dim)
        else:
            self.projector = nn.Identity()

        if predictor:
            self.predictor = build_mlp(2, 256, 4096, 256, last_bn=False)
        else:
            self.predictor = nn.Identity()

    def forward(
        self,
        x: Union[Tensor, List[Tensor]],
        length: List[Tensor],
        mask: List[Tensor],
        mask_input: Tensor,
    ) -> Tensor:
        """
        Applies the encoder and projection/prediction head to input.

        Parameters
        ----------
        x : Union[Tensor, List[Tensor]]
            Input tensor or list of tensors corresponding to multiple crops.
        length : List[Tensor]
            List of tensors representing the valid lengths of input sequences.
        mask : List[Tensor]
            List of tensors representing masking indices for the encoder.
        mask_input : Tensor
            Tensor representing additional masking information for the encoder.

        Returns
        -------
        Tensor
        """
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx, output_frame, _ = (
            0,
            torch.empty(0).to(x[0].device),
            torch.empty(0).to(x[0].device),
        )

        for end_idx in idx_crops:
            _out_frame = self.encoder(
                torch.cat(x[start_idx:end_idx]),
                length=torch.cat(length[start_idx:end_idx]),
                mask_index=torch.cat(mask[start_idx:end_idx]),
                mask_input=mask_input,
            )
            # accumulate outputs
            output_frame = torch.cat((output_frame, _out_frame))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.predictor(self.projector(output_frame))


class FrameATST(nn.Module):
    """See original frame-atst package for documentation"""

    def __init__(
        self,
        arch: str = "small",
        symmetric: bool = True,
        pos_type: str = "cut",
        avg_blocks: int = 0,
        patch_embed: str = "Linear",
        **kwargs: object,
    ) -> None:
        """
        Initializes the BYOL-based model with student and teacher encoders.

        Raises
        ------
        RuntimeError
            If `arch` is not one of {"small", "base"}.
        """
        super().__init__()
        if arch == "small":
            encoder_fn = FrameAST_small
            embed_dim = 384
        elif arch == "base":
            encoder_fn = FrameAST_base
            embed_dim = 768
        else:
            raise RuntimeError("arch {} is not implemented".format(arch))
        self.symmetric = symmetric
        if avg_blocks == 0:  # atst-frame
            self.student = MultiCropWrapper(
                encoder_fn(pos_type=pos_type, patch_embed=patch_embed, **kwargs),
                embed_dim,
                predictor=True,
            )
            self.teacher = MultiCropWrapper(
                encoder_fn(pos_type=pos_type, patch_embed=patch_embed, **kwargs),
                embed_dim,
                predictor=False,
            )
        else:  # data2vec, no projector, predictor is linear
            self.student = MultiCropWrapper(
                encoder_fn(pos_type=pos_type, patch_embed=patch_embed, **kwargs),
                embed_dim,
                projector="linear",
                predictor=False,
            )
            self.teacher = MultiCropWrapper(
                encoder_fn(
                    pos_type=pos_type, patch_embed=patch_embed, avg_blocks=8, **kwargs
                ),
                embed_dim,
                projector=None,
                predictor=False,
            )
        for p in self.teacher.parameters():
            p.requires_grad = False

        if avg_blocks == 0:  # atst-frame
            self.teacher.load_state_dict(
                {
                    k: v
                    for k, v in self.student.state_dict().items()
                    if "predictor" not in k
                }
            )
        else:  # data2vec
            self.teacher.load_state_dict(
                {
                    k: v
                    for k, v in self.student.state_dict().items()
                    if "projector" not in k
                }
            )

        self.loss_fn = ByolLoss(symmetric=symmetric)

    def forward(
        self, x: Union[Tensor, List[Tensor]], length: List[Tensor], mask: List[Tensor]
    ) -> Tensor:
        """
        Computes the BYOL loss between student and teacher model outputs

        Returns
        -------
        Tensor
        """
        if self.symmetric:
            tea = self.teacher(x, length, mask, False)
            stu = self.student(x, length, mask, True)
            return self.loss_fn(stu, tea)
        else:
            tea = self.teacher(x[:1], length[:1], mask[:1], False)
            stu = self.student(x[1:], length[1:], mask[1:], True)
            return self.loss_fn(stu, tea)

    def update_teacher(self, m: Union[np.ndarray, float]) -> None:
        """See original frame-atst package for documentation"""
        with torch.no_grad():
            for param_q, param_k in zip(
                self.student.encoder.parameters(),
                self.teacher.encoder.parameters(),
                strict=False,
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(
                self.student.projector.parameters(),
                self.teacher.projector.parameters(),
                strict=False,
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def _init_teacher(self) -> None:
        """See original frame-atst package for documentation"""
        self.teacher.load_state_dict(
            {k: v for k, v in self.student.state_dict().items() if "predictor" not in k}
        )


def cosine_scheduler_step(
    base_value: float,
    final_value: float,
    max_steps: int,
    warmup_steps: int = 0,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    """
    Generates a cosine learning rate schedule with optional warmup phase.

    Returns
    -------
    np.ndarray
        Array of length `max_steps` containing the scheduled learning rates.
    """
    warmup_schedule = np.array([])
    if warmup_steps > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_steps)

    iters = np.arange(max_steps - warmup_steps)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == max_steps
    return schedule


def get_params_groups(model: nn.Module) -> List:
    """See original frame-atst package for documentation

    Returns
    -----------
    List
    """
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def bool_flag(s: str) -> bool:
    """
    Parse boolean arguments from the command line.

    Returns
    -------
    bool

    Raises
    ------
    argparse.ArgumentTypeError
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


class FrameATSTLightningModule(LightningModule):
    """See original frame-atst package for documentation"""

    def __init__(
        self,
        arch: str = "small",
        learning_rate: float = 5e-4,
        warmup_steps: int = 1300,
        max_steps: int = 39000,
        ema: float = 0.99,
        symmetric: bool = True,
        pos_type: str = "cut",
        avg_blocks: int = 0,
        patch_embed: str = "Linear",
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.model = FrameATST(
            arch=arch,
            symmetric=symmetric,
            pos_type=pos_type,
            avg_blocks=avg_blocks,
            patch_embed=patch_embed,
            **kwargs,
        )
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.symmetric = symmetric
        self.ema_scheduler = cosine_scheduler_step(ema, 1, max_steps, 0)
        self.wd_scheduler = cosine_scheduler_step(0.04, 0.4, max_steps, 0)
        self.mylr_scheduler = cosine_scheduler_step(
            learning_rate, 1e-6, max_steps, warmup_steps
        )
        self.save_hyperparameters()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """See original frame-atst package for documentation

        Returns
        ---------
        torch.Tensor
        """
        self.schedule()
        (melspecs, lengths, masks), _ = batch
        total_loss_frm, std_frm_stu, std_frm_tea = self.model(melspecs, lengths, masks)
        loss = total_loss_frm
        self.log("loss", loss, prog_bar=True, logger=True)
        self.log("loss_frm", total_loss_frm, prog_bar=True, logger=True)
        self.log("std_frm_tea", std_frm_tea, prog_bar=True, logger=True)
        self.log("std_frm_stu", std_frm_stu, prog_bar=True, logger=True)
        self.log(
            "ema", self.ema_scheduler[self.global_step], prog_bar=True, logger=True
        )
        self.log("step", self.global_step, prog_bar=True, logger=True)

        return loss

    def freeze(self) -> None:
        """See original frame-atst package for documentation"""
        return super().freeze()

    def unfreeze(self) -> None:
        """See original frame-atst package for documentation"""
        return super().unfreeze()

    def schedule(self) -> None:
        """See original frame-atst package for documentation"""
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            param_group["lr"] = self.mylr_scheduler[self.global_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_scheduler[self.global_step]

        self.log("wd", self.wd_scheduler[self.global_step], prog_bar=True, logger=True)
        self.log("lr", param_group["lr"], prog_bar=True, logger=True)

    def configure_optimizers(self) -> None:
        """See original frame-atst package for documentation"""
        pass
        # optimizer = AdamW(get_params_groups(self.model.student),
        #                   lr=self.learning_rate,
        #                   weight_decay=0.)
        # return [optimizer]

    def on_train_batch_end(
        self,
        outputs: torch.Tensor,
        batch: torch.Tensor,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """See original frame-atst package for documentation"""
        m = self.ema_scheduler[self.global_step]
        self.model.update_teacher(m)

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """See original frame-atst package for documentation

        Returns
        ------
        argparse.ArgumentParser
        """
        parser = parent_parser.add_argument_group("FrameATSTModel")
        parser.add_argument("--arch", type=str, default="small")
        parser.add_argument(
            "--symmetric",
            type=bool_flag,
            default=True,
            help="whether to use symemtric loss",
        )
        parser.add_argument(
            "--nprompt",
            type=int,
            default=0,
            help="number of prompts, not used, always 0",
        )
        parser.add_argument(
            "--learning_rate",
            default=0.0005,
            type=float,
            help="""Learning rate at the end of
            linear warmup (highest LR used during training).
            The learning rate is linearly scaled
            with the batch size, and specified here for a
            reference batch size of 256.""",
        )
        parser.add_argument(
            "--ema",
            default=0.99,
            type=float,
            help="""Base EMA
            parameter for teacher update. The value is
              increased to 1 during training with cosine schedule.
            """,
        )
        parser.add_argument("--warmup_steps", default=1300, type=int)
        parser.add_argument("--max_steps", default=39010, type=int)
        parser.add_argument(
            "--pos_type",
            default="cut",
            type=str,
            help="""
            'cut' denotes absolute positional embedding,
            'interpolate' denotes 2D positional embedding used in SSAST""",
        )
        parser.add_argument(
            "--avg_blocks",
            default=0,
            type=int,
            help="0 means atst-frame, a positive int value means data2vec style loss",
        )
        parser.add_argument(
            "--patch_embed",
            default="Linear",
            type=str,
            help="Linear or CNN patch embedding",
        )
        return parent_parser


def FrameAST_small(patch_h: int = 64, patch_w: int = 4, **kwargs: object) -> nn.Module:
    """See original frame-atst package for documentation

    Returns
    -------
    nn.Module
    """
    return FrameAST(
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def FrameAST_base(patch_h: int = 64, patch_w: int = 4, **kwargs: object) -> nn.Module:
    """See original frame-atst package for documentation

    Returns
    ---------
    nn.Module
    """
    return FrameAST(
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def FrameAST_large(patch_h: int, patch_w: int, **kwargs: object) -> nn.Module:
    """See original frame-atst package for documentation

    Returns
    ---------
    nn.Module
    """
    return FrameAST(
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def load_model(model_path: str) -> LightningModule:
    """See original frame-atst package for documentation

    Returns
    --------
        LightningModule
    """
    melspec_t = torchaudio.transforms.MelSpectrogram(
        16000,
        f_min=60,
        f_max=7800,
        hop_length=160,
        win_length=1024,
        n_fft=1024,
        n_mels=64,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    normalize = MinMax(min=-79.6482, max=50.6842)
    s = torch.load(model_path)

    pretrained_model = FrameATSTLightningModule.load_from_checkpoint(model_path)

    pretrained_encoder = pretrained_model.model.teacher.encoder
    pretrained_encoder.hyper_param = s["hyper_parameters"]

    pretrained_encoder.sample_rate = 16000
    pretrained_encoder.scene_embedding_size = (
        pretrained_encoder.embed_dim * 2 * N_BLOCKS
    )
    pretrained_encoder.timestamp_embedding_size = (
        pretrained_encoder.embed_dim * N_BLOCKS
    )

    pretrained_encoder.train()
    pretrained_encoder.transform = transforms.Compose([melspec_t, to_db, normalize])

    return pretrained_encoder


def get_scene_embedding(audio: torch.Tensor, model: LightningModule) -> torch.Tensor:
    """
    extract scene (clip-level) embedding from an audio clip
    =======================================
    args:
        audio: torch.tensor in the shape of [1,N] or [B,1,N]
        model: the pretrained encoder returned by load_model
    Returns
    ---------
        emb: returned embedding in the shape of [1,N_BLOCKS*emb_size]
        or [B,N_BLOCKS*emb_size], where emb_size is 768 for base model
        and 384 for small model.

    """
    if len(audio.shape) == 2:
        audio = audio.unsqueeze(1)
    else:
        assert len(audio.shape) == 3

    model.to(audio.device)
    model.transform.transforms[0].to(audio.device)
    mel = model.transform(audio)
    # length = torch.tensor([mel.shape[-1]]).expand(mel.shape[0])
    chunk_len = 1001  # 10 secnods, consistent with the length of positional embedding
    total_len = mel.shape[-1]
    num_chunks = total_len // chunk_len + 1
    output = []
    for i in range(num_chunks):
        start = i * chunk_len
        end = (i + 1) * chunk_len
        if end > total_len:
            end = total_len
        if end > start:  # and (length +chunk_len//2  > end):
            mel_chunk = mel[:, :, :, start:end]
            len_chunk = mel_chunk.shape[
                -1
            ]  # if length>end+chunk_len else (length - end)
            len_chunk = torch.tensor([len_chunk]).expand(mel.shape[0]).to(audio.device)
            output_chunk = model.get_intermediate_layers(mel_chunk, len_chunk, n=12)

            output.append(output_chunk)
    output = torch.stack(output, dim=0)
    output = torch.mean(output, dim=0)

    return output


def get_timestamp_embedding(
    audio: torch.Tensor, model: LightningModule
) -> torch.Tensor:
    """
    Extract frame-level embeddings from an audio clip
    ==================================================
    args:
        audio: torch.tensor in the shape of [1,N] or [B,1,N]
        model: the pretrained encoder returned by load_model
    Returns
    -----------
        emb: returned embedding in the shape of
            [1,T,N_BLOCKS*emb_size] or [B,T,N_BLOCKS,emb_size],
            where emb_size is 768 for base model and 384 for small model.
    """
    if len(audio.shape) == 2:
        audio = audio.unsqueeze(1)
    else:
        assert len(audio.shape) == 3

    model.to(audio.device)
    model.transform.transforms[0].to(audio.device)
    mel = model.transform(audio)
    output = []

    chunk_len = 1001  # 10 secnods, consistent with the length of positional embedding

    total_len = mel.shape[-1]
    num_chunks = total_len // chunk_len + 1
    for i in range(num_chunks):
        start = i * chunk_len
        end = (i + 1) * chunk_len
        if end > total_len:
            end = total_len
        if end > start:
            mel_chunk = mel[:, :, :, start:end]
            len_chunk = (
                torch.tensor([mel_chunk.shape[-1]])
                .expand(mel.shape[0])
                .to(audio.device)
            )

            output_chunk = model.get_intermediate_layers(
                mel_chunk, len_chunk, n=N_BLOCKS, scene=False
            )

            output.append(output_chunk)
    output = torch.cat(output, dim=1)
    # length = output.shape[1]
    # timestamps = (
    #     (torch.arange(length) * 40).float().unsqueeze(0).expand(mel.shape[0], -1)
    # )
    return output.permute(0, 2, 1)
