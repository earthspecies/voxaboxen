import math
import numpy as np
import pandas as pd
import librosa

from numpy.random import default_rng
from intervaltree import IntervalTree
from torch.utils.data import Dataset, DataLoader

import torch
import torchaudio
from torch.nn import functional as F

def normalize_sig_np(sig, eps=1e-8):
    """
    Normalize a signal to [-1, 1] range.

    Parameters
    ----------
    sig : numpy.ndarray
        Input signal to normalize
    eps : float, optional
        Small constant to avoid division by zero, by default 1e-8

    Returns
    -------
    numpy.ndarray
        Normalized signal
    """

    sig = sig / (np.max(np.abs(sig))+eps)
    return sig

def crop_and_pad(wav, sr, dur_sec):
    """
    Crop or pad waveform to match target duration and sample rate.

    Used after resampling when loading data to ensure the waveform is the
    proper size for the dataset.

    Parameters
    ----------
    wav : torch.Tensor
        Input waveform tensor
    sr : int
        Sample rate
    dur_sec : float
        Target duration in seconds

    Returns
    -------
    torch.Tensor
        Waveform with exact target duration in samples
    """

    target_dur_samples = int(sr * dur_sec)
    wav = wav[..., :target_dur_samples]

    pad = target_dur_samples - wav.size(-1)
    if pad > 0:
        wav = F.pad(wav, (0,pad)) #padding starts from last dims

    return wav

class DetectionDataset(Dataset):
    """
    PyTorch Dataset for loading audio segments and corresponding annotations
    from across multiple files.

    Parameters
    ----------
    info_df : pandas.DataFrame
        DataFrame containing audio file metadata with columns:
        - 'fn': filename identifier
        - 'audio_fp': audio file path
        - 'selection_table_fp': annotation file path
    train : bool
        Whether dataset is for training
    args : argparse.Namespace
        Configuration arguments containing:
        - label_set: List of class labels
        - unknown_label: Label for unknown classes
        - label_mapping: Dictionary mapping annotation labels to model classes
        - sr: Sample rate
        - clip_duration: Duration of audio clips in seconds
        - clip_hop: Hop size between clips in seconds
        - seed: Random seed
        - scale_factor: Downsampling factor
        - stereo/multichannel: Audio channel configuration
        - omit_empty_clip_prob: Probability of omitting empty clips during training
    random_seed_shift : int, optional
        Additional seed offset, by default 0
    """

    def __init__(self, info_df, train, args, random_seed_shift = 0):
        self.info_df = info_df
        self.label_set = args.label_set
        if hasattr(args, 'unknown_label'):
            self.unknown_label = args.unknown_label
        else:
            self.unknown_label = None
        self.label_mapping = args.label_mapping
        self.n_classes = len(self.label_set)
        self.sr = args.sr
        self.clip_duration = args.clip_duration
        self.clip_hop = args.clip_hop
        assert (self.clip_hop*args.sr).is_integer()
        self.seed = args.seed + random_seed_shift

        self.scale_factor = args.scale_factor
        #self.prediction_scale_factor = args.prediction_scale_factor
        #self.scale_factor_raw_to_prediction = self.scale_factor*self.prediction_scale_factor
        self.rng = default_rng(seed=self.seed)
        self.train=train
        if hasattr(args, 'stereo') and args.stereo:
            self.mono = False
        elif hasattr(args, 'multichannel') and args.multichannel:
            self.mono = False
        else:
            self.mono = True

        if self.train:
            self.omit_empty_clip_prob = args.omit_empty_clip_prob
            self.clip_start_offset = self.rng.integers(0, np.floor(self.clip_hop*self.sr)) / self.sr
        else:
            self.omit_empty_clip_prob = 0
            self.clip_start_offset = 0

        self.args=args
        # make metadata
        self.make_metadata()

    def process_selection_table(self, selection_table_fp):
        """
        Process annotation file into interval tree format.

        Parameters
        ----------
        selection_table_fp : str
            Path to annotation file (tab-separated)

        Returns
        -------
        IntervalTree
            Tree containing labeled time intervals
        """

        selection_table = pd.read_csv(selection_table_fp, sep = '\t')
        tree = IntervalTree()

        for ii, row in selection_table.iterrows():
            start = row['Begin Time (s)']
            end = row['End Time (s)']
            label = row['Annotation']

            if end<=start:
                continue

            if label in self.label_mapping:
                label = self.label_mapping[label]
            else:
                continue

            if label == self.unknown_label:
                label_idx = -1
            else:
                label_idx = self.label_set.index(label)
            tree.addi(start, end, label_idx)

        return tree

    def make_metadata(self):
        """Generate dataset metadata including clip boundaries."""

        selection_table_dict = dict()
        metadata = []

        for ii, row in self.info_df.iterrows():
            fn = row['fn']
            audio_fp = row['audio_fp']

            duration = librosa.get_duration(path=audio_fp)
            selection_table_fp = row['selection_table_fp']

            selection_table = self.process_selection_table(selection_table_fp)
            selection_table_dict[fn] = selection_table

            num_clips = max(0, int(np.floor((duration - self.clip_duration - self.clip_start_offset) // self.clip_hop)))

            for tt in range(num_clips):
                start = tt*self.clip_hop + self.clip_start_offset
                end = start + self.clip_duration

                ivs = selection_table[start:end]
                # if no annotated intervals, skip with specified probability
                if not ivs:
                  if self.omit_empty_clip_prob > self.rng.uniform():
                      continue

                metadata.append([fn, audio_fp, start, end])

        self.selection_table_dict = selection_table_dict
        self.metadata = metadata

    def get_pos_intervals(self, fn, start, end):
        """
        Get annotated intervals within specified time range.

        Parameters
        ----------
        fn : str
            Filename identifier
        start : float
            Start time in seconds
        end : float
            End time in seconds

        Returns
        -------
        list
            List of (start, end, label_idx) tuples
        """

        tree = self.selection_table_dict[fn]

        intervals = tree[start:end]
        intervals = [(max(iv.begin, start)-start, min(iv.end, end)-start, iv.data) for iv in intervals]

        return intervals

    def get_class_proportions(self):
        """
        Calculate class distribution in dataset.

        Returns
        -------
        numpy.ndarray
            Array of class proportions
        """

        counts = np.zeros((self.n_classes,))

        for k in self.selection_table_dict:
            st = self.selection_table_dict[k]
            for interval in st:
                annot = interval.data
                if annot == -1:
                    continue
                else:
                    counts[annot] += 1

        total_count = np.sum(counts)
        proportions = counts / total_count

        return proportions

    def get_annotation(self, pos_intervals, audio):
        """
        Generate target annotations from positive intervals.

        Parameters
        ----------
        pos_intervals : list
            List of (start, end, label_idx) tuples
        audio : torch.Tensor
            Input audio tensor

        Returns
        -------
        tuple
            Tuple containing:
            - anchor_annos: Anchor point annotations
            - regression_annos: Duration annotations
            - class_annos: Class probability annotations
            - rev_anchor_annos: Reverse anchor points
            - rev_regression_annos: Reverse duration
            - rev_class_annos: Reverse class probabilities
        """

        raw_seq_len = audio.shape[-1]
        seq_len = int(math.ceil(raw_seq_len / self.scale_factor))
        anno_sr = int(self.sr // self.scale_factor)

        regression_annos = np.zeros((seq_len,))
        class_annos = np.zeros((seq_len, self.n_classes))
        anchor_annos = [np.zeros(seq_len,)]
        rev_regression_annos = np.zeros((seq_len,))
        rev_class_annos = np.zeros((seq_len, self.n_classes))
        rev_anchor_annos = [np.zeros(seq_len,)]

        for iv in pos_intervals:
            start, end, class_idx = iv
            dur = end-start
            dur_samples = np.ceil(dur * anno_sr)

            start_idx = int(math.floor(start*anno_sr))
            start_idx = max(min(start_idx, seq_len-1), 0)

            end_idx = int(math.ceil(end*anno_sr))
            end_idx = max(min(end_idx, seq_len-1), 0)
            dur_samples = int(np.ceil(dur * anno_sr))

            anchor_anno = get_anchor_anno(start_idx, dur_samples, seq_len)
            anchor_annos.append(anchor_anno)
            regression_annos[start_idx] = dur

            rev_anchor_anno = get_anchor_anno(end_idx, dur_samples, seq_len)
            rev_anchor_annos.append(rev_anchor_anno)
            rev_regression_annos[end_idx] = dur

            if hasattr(self.args,"segmentation_based") and self.args.segmentation_based:
                if class_idx == -1:
                    pass
                else:
                    class_annos[start_idx:start_idx+dur_samples,class_idx]=1.

            else:
                if class_idx != -1:
                    class_annos[start_idx, class_idx] = 1.
                    rev_class_annos[end_idx, class_idx] = 1.
                else:
                    class_annos[start_idx, :] = 1./self.n_classes # if unknown, enforce uncertainty
                    rev_class_annos[end_idx, :] = 1./self.n_classes # if unknown, enforce uncertainty


        anchor_annos = np.stack(anchor_annos)
        anchor_annos = np.amax(anchor_annos, axis = 0)
        rev_anchor_annos = np.stack(rev_anchor_annos)
        rev_anchor_annos = np.amax(rev_anchor_annos, axis = 0)
        # shapes [time_steps, ], [time_steps, ], [time_steps, n_classes] (times two)
        return anchor_annos, regression_annos, class_annos, rev_anchor_annos, rev_regression_annos, rev_class_annos

    def __getitem__(self, index):
        """
        Get dataset item by index.

        Parameters
        ----------
        index : int
            Item index

        Returns
        -------
        tuple
            Tuple containing:
            - audio: Audio tensor
            - anchor_anno: Anchor annotations
            - regression_anno: Duration annotations
            - class_anno: Class annotations
            - rev_anchor_anno: Reverse anchors
            - rev_regression_anno: Reverse durations
            - rev_class_anno: Reverse classes
        """

        fn, audio_fp, start, end = self.metadata[index]

        audio, file_sr = librosa.load(audio_fp, sr=None, offset=start, duration=self.clip_duration, mono=self.mono)
        audio = torch.from_numpy(audio)

        audio = audio-torch.mean(audio, -1, keepdim=True)
        if file_sr != self.sr:
            audio = torchaudio.functional.resample(audio, file_sr, self.sr)

        audio = crop_and_pad(audio, self.sr, self.clip_duration)

        pos_intervals = self.get_pos_intervals(fn, start, end)
        anchor_anno, regression_anno, class_anno, rev_anchor_anno, rev_regression_anno, rev_class_anno = self.get_annotation(pos_intervals, audio)

        return audio, torch.from_numpy(anchor_anno), torch.from_numpy(regression_anno), torch.from_numpy(class_anno), torch.from_numpy(rev_anchor_anno), torch.from_numpy(rev_regression_anno), torch.from_numpy(rev_class_anno)

    def __len__(self):
        return len(self.metadata)


def get_train_dataloader(args, random_seed_shift = 0):
    """
    Create training DataLoader.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration arguments
    random_seed_shift : int, optional
        Additional seed offset, by default 0

    Returns
    -------
    torch.DataLoader
    """

    train_info_fp = args.train_info_fp
    train_info_df = pd.read_csv(train_info_fp)

    train_dataset = DetectionDataset(train_info_df, True, args, random_seed_shift = random_seed_shift)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last = True)

    return train_dataloader


class SingleClipDataset(Dataset):
    """
    PyTorch Dataset for loading audio segments from a single file.

    Dataset for processing single audio clips.

    Parameters
    ----------
    audio_fp : str
        Path to audio file
    clip_hop : float
        Hop size between clips in seconds
    args : argparse.Namespace
        Configuration arguments
    annot_fp : str, optional
        Path to annotation file, by default None
    """

    def __init__(self, audio_fp, clip_hop, args, annot_fp = None):
        # waveform (samples,)
        super().__init__()
        self.duration = librosa.get_duration(path=audio_fp)
        self.clip_hop = clip_hop
        self.num_clips = int(np.ceil(self.duration / self.clip_hop)) #max(0, int(np.floor(self.duration / self.clip_hop)+1)) #int(np.floor((self.duration - args.clip_duration) // clip_hop))
        self.audio_fp = audio_fp
        self.clip_duration = args.clip_duration
        self.annot_fp = annot_fp # attribute that is accessed elsewhere
        self.sr = args.sr
        if hasattr(args, 'stereo') and args.stereo:
            self.mono = False
        elif hasattr(args, 'multichannel') and args.multichannel:
            self.mono = False
        else:
            self.mono = True

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        """Get audio clip by index and return as torch.tensor."""
        start = idx * self.clip_hop

        audio, file_sr = librosa.load(self.audio_fp, sr=None, offset=start, duration=self.clip_duration, mono=self.mono)
        audio = torch.from_numpy(audio)


        audio = audio-torch.mean(audio, -1, keepdim=True)
        if file_sr != self.sr:
            audio = torchaudio.functional.resample(audio, file_sr, self.sr)

        audio = crop_and_pad(audio, self.sr, self.clip_duration)

        return audio

def get_single_clip_data(audio_fp, clip_hop, args, annot_fp = None):
    """
    Create DataLoader for single audio file.

    Parameters
    ----------
    audio_fp : str
        Path to audio file
    clip_hop : float
        Hop size between clips in seconds
    args : argparse.Namespace
        Configuration arguments
    annot_fp : str, optional
        Path to annotation file, by default None

    Returns
    -------
    DataLoader
        Single clip DataLoader
    """

    return DataLoader(
        SingleClipDataset(audio_fp, clip_hop, args, annot_fp = annot_fp),
        batch_size = args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

def get_val_dataloader(args):
    """
    Create validation DataLoaders.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration arguments

    Returns
    -------
    dict
        Dictionary mapping filenames to DataLoaders
    """

    val_info_fp = args.val_info_fp
    val_info_df = pd.read_csv(val_info_fp)

    val_dataloaders = {}

    for i in range(len(val_info_df)):
        fn = val_info_df.iloc[i]['fn']
        audio_fp = val_info_df.iloc[i]['audio_fp']
        annot_fp = val_info_df.iloc[i]['selection_table_fp']
        val_dataloaders[fn] = get_single_clip_data(audio_fp, args.clip_duration/2, args, annot_fp = annot_fp)

    return val_dataloaders

def get_test_dataloader(args):
    """
    Create test DataLoaders.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration arguments

    Returns
    -------
    dict
        Dictionary mapping filenames to DataLoaders
    """

    test_info_fp = args.test_info_fp
    test_info_df = pd.read_csv(test_info_fp)

    test_dataloaders = {}

    for i in range(len(test_info_df)):
        fn = test_info_df.iloc[i]['fn']
        audio_fp = test_info_df.iloc[i]['audio_fp']
        annot_fp = test_info_df.iloc[i]['selection_table_fp']
        test_dataloaders[fn] = get_single_clip_data(audio_fp, args.clip_duration/2, args, annot_fp = annot_fp)

    return test_dataloaders

def get_anchor_anno(start_idx, dur_samples, seq_len):
    """
    Represent start idx as a Gaussian blurred onehot encoding.

        Parameters
    ----------
    start_idx : int
        Start index of annotation
    dur_samples : int
        Duration in samples
    seq_len : int
        Total sequence length

    Returns
    -------
    numpy.ndarray
        Anchor point annotations

    Notes
    -----
    This setting of `std` follows CornerNet, where adaptive standard deviation
    is set to 1/3 image radius.

    """

    std = dur_samples / 6
    x = (np.arange(seq_len) - start_idx) ** 2
    x = x / (2 * std**2)
    x = np.exp(-x)
    return x
