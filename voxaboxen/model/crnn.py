"""
Convolutional-recurrent neural network for extracting audio features
"""

import torch
import torchaudio
from einops import rearrange
from torch import nn


class CRNN(nn.Module):
    """
    Convolutional-recurrent neural network for extracting audio features
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.downsample_factor = self.args.scale_factor
        device = "cuda" if torch.cuda.is_available() else "cpu"

        n_mels = 256
        hidden_size = 64
        rnn_hidden_size = self.args.rnn_hidden_size
        n_blocks = 1

        self.output_dim = 2 * rnn_hidden_size

        self.spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.args.sr,
            n_fft=n_mels * 6,
            hop_length=self.downsample_factor // 2,
            n_mels=n_mels,
        ).to(device)

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(
            1, hidden_size, kernel_size=7, stride=1, padding="same", bias=False
        ).to(device)
        self.bn1 = nn.BatchNorm2d(hidden_size).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)

        self.pool1 = nn.AdaptiveAvgPool2d((n_mels // 2, None)).to(device)

        # Residual block 1
        self.resblock1 = []
        for _ in range(n_blocks):
            self.resblock1.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_size,
                        hidden_size,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        hidden_size,
                        hidden_size,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_size),
                )
            )

        self.resblock1 = nn.ModuleList(self.resblock1).to(device)

        self.pool2 = nn.AdaptiveAvgPool2d((n_mels // 4, None)).to(device)

        # Residual block 2
        self.resblock2 = []
        for _ in range(n_blocks):
            self.resblock2.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_size,
                        hidden_size,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        hidden_size,
                        hidden_size,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_size),
                )
            )

        self.resblock2 = nn.ModuleList(self.resblock2).to(device)

        self.pool3 = nn.AdaptiveAvgPool2d((n_mels // 8, None)).to(device)

        self.pool3b = nn.AvgPool1d(2).to(device)  ###

        self.head = nn.LSTM(
            hidden_size * (n_mels // 8),
            rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        ).to(device)

    def freeze(self):
        """
        Freeze audio encoder
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze audio encoder
        """
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, audio):
        """
        Forward pass
        Parameters
        ----------
        audio : torch.Tensor
            audio of shape [batch, time]
        Returns
        -------
        torch.Tensor
            features of shape [batch, time, channels]
        """
        expected_output_dur = audio.size(1) // self.downsample_factor

        x = self.spectrogram(audio)  # b c t
        x = torch.log(x + torch.full_like(x, 1e-10))

        x = x.unsqueeze(1)  # b 1 c t

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        for b in self.resblock1:
            x = b(x) + x
        x = self.pool2(x)

        for b in self.resblock2:
            x = b(x) + x
        x = self.pool3(x)  # b 64 16 t

        x = torch.reshape(x, (x.size(0), -1, x.size(-1)))
        x = self.pool3b(x)

        #
        pad = expected_output_dur - x.size(-1)
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad), mode="reflect")
        x = x[:, :, :expected_output_dur]
        #

        x = rearrange(x, "b c t -> b t c")
        x = self.head(x)[0]

        return x
