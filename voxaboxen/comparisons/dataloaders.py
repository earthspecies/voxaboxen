import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from detectron2.engine import DefaultTrainer
from detectron2.structures import Instances, Boxes
from intervaltree import IntervalTree

from voxaboxen.data.data import DetectionDataset, SingleClipDataset, crop_and_pad

def get_torch_mel_frequencies(f_max, n_mels, f_min=0.0, mel_scale="htk"):
    m_min = torchaudio.functional.functional._hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = torchaudio.functional.functional._hz_to_mel(f_max, mel_scale=mel_scale)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = torchaudio.functional.functional._mel_to_hz(m_pts, mel_scale=mel_scale)
    return f_pts

class DetectronDataset(DetectionDataset):
    def __init__(self, detectron_cfg, info_df, train, args, random_seed_shift = 0, collect_statistics=False):
        super().__init__(info_df, train, args, random_seed_shift)
        self.spectrogram_args = detectron_cfg.SPECTROGRAM
        f_max = float(self.sr // 2)
        self.make_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.sr,
            f_min = self.spectrogram_args.F_MIN,
            f_max = f_max,
            n_fft = self.spectrogram_args.N_FFT,
            win_length = self.spectrogram_args.WIN_LENGTH,
            hop_length = self.spectrogram_args.HOP_LENGTH,
            n_mels = self.spectrogram_args.N_MELS,
            )
        self.spectrogram_t = lambda n_frames: (np.arange(n_frames)*self.spectrogram_args.HOP_LENGTH)/self.sr
        self.spectrogram_f = get_torch_mel_frequencies(f_max=f_max, f_min=self.spectrogram_args.F_MIN, n_mels=self.spectrogram_args.N_MELS).numpy()[1:-1] # Using defaults
        self.collect_statistics = collect_statistics
        self.mixup = args.mixup and train

    def process_selection_table(self, selection_table_fp):
        selection_table = pd.read_csv(selection_table_fp, sep = '\t')
        tree = IntervalTree()

        for _, row in selection_table.iterrows():
            start = row['Begin Time (s)']
            end = row['End Time (s)']
            bottom = row['Low Freq (Hz)']
            top = row['High Freq (Hz)']
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

            tree.addi(start, end, (bottom, top, label_idx))

        return tree

    def get_pos_intervals(self, fn, start, end):
        tree = self.selection_table_dict[fn]
        intervals = tree[start:end]

        intervals = [(max(iv.begin, start)-start, min(iv.end, end)-start, iv.data[0], iv.data[1], iv.data[2]) for iv in intervals]
        return intervals

    def convert_intervals_to_boxes(self, intervals, n_time_frames):
        """ Convert intervals to [tstart, fstart, tstop, fstop] format, where the elements are indexes"""
        t = self.spectrogram_t(n_time_frames)
        f = self.spectrogram_f
        aa = lambda x, x0: np.argmin(np.abs(x - x0))
        index_intervals = []
        for iv in intervals:
            onset_idx = aa(iv[0], t)
            low_f_idx = aa(iv[2], f)
            offset_idx = aa(iv[1], t)
            high_f_idx = aa(iv[3], f)
            # Prevent dimension from collapsing
            if onset_idx == offset_idx:
                if onset_idx == len(t) - 1:
                    onset_idx -= 1
                else:
                    offset_idx += 1
            if low_f_idx == high_f_idx:
                if low_f_idx == len(f) - 1:
                    low_f_idx -= 1
                else:
                    high_f_idx += 1
            index_intervals.append([onset_idx, low_f_idx, offset_idx, high_f_idx])
        return index_intervals

    def power_to_dB(self, mel_spectrogram):
        return torchaudio.functional.amplitude_to_DB(mel_spectrogram[None, None, :, :], multiplier=10., amin=self.spectrogram_args.REF, db_multiplier=np.log10(self.spectrogram_args.REF)).squeeze()

    def spectrogram_to_image(self, spectrogram):
        """Add channels to basic spectrogram (in dB) to make up for differences with (visual) images"""
        img = spectrogram[None, :, :] #Should be channel=1, f, t

        mask = (img > self.spectrogram_args.FLOOR_THRESHOLD).to(img) #TODO: May not be helpful.
        img = (img - self.spectrogram_args.FLOOR_THRESHOLD) * mask
        img = (img / self.spectrogram_args.CEIL_THRESHOLD)*255.0

        boundaries = torch.ones(img.shape, dtype=img.dtype).to(img)
        full_img = torch.cat((img, mask, boundaries), dim=0) #channels=3, freqs, time

        return full_img.to(torch.float32)

    def __getitem__(self, index):
        fn, audio_fp, start, end = self.metadata[index]
        audio, file_sr = librosa.load(audio_fp, sr=None, offset=start, duration=self.clip_duration, mono=True)
        audio = audio-np.mean(audio)
        if self.amp_aug and self.train:
            audio = self.augment_amplitude(audio)
        audio = torch.from_numpy(audio)
        if file_sr != self.sr:
          audio = torchaudio.functional.resample(audio, file_sr, self.sr)
          audio = crop_and_pad(audio, self.sr, self.clip_duration)

        pos_intervals = self.get_pos_intervals(fn, start, end)
        record = {"sound_name": fn}
        mel_spectrogram = self.make_mel_spectrogram(audio) # size: (channel if any, n_mels, time)
        if self.collect_statistics:
           record["power"] = mel_spectrogram
        if self.mixup:
            # Save audio so it can be added to other datapoint's audio later
            record["audio"] = audio
        mel_spectrogram_dB = self.power_to_dB(mel_spectrogram)
        record["image"] = self.spectrogram_to_image(mel_spectrogram_dB)
        record["height"] = mel_spectrogram.shape[0] #f
        record["width"] = mel_spectrogram.shape[1] #t

        # -- Create annotations ---
        record["instances"] = Instances(mel_spectrogram.shape)
        record["instances"].gt_classes = torch.LongTensor(np.array([pi[-1] for pi in pos_intervals]))
        #record["instances"].gt_ibm = BitMasks(masks.permute(0,2,1).contiguous())
        boxes = self.convert_intervals_to_boxes(pos_intervals, mel_spectrogram.shape[1])
        record["instances"].gt_boxes = Boxes(boxes)

        return record

class DetectronSingleClipDataset(SingleClipDataset):
    def __init__(self, detectron_cfg, audio_fp, clip_hop, args, annot_fp=None):
        super().__init__(audio_fp, clip_hop, args, annot_fp)
        self.spectrogram_args = detectron_cfg.SPECTROGRAM
        self.make_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.sr,
            f_min = self.spectrogram_args.F_MIN,
            f_max = self.spectrogram_args.F_MAX,
            n_fft = self.spectrogram_args.N_FFT,
            win_length = self.spectrogram_args.WIN_LENGTH,
            hop_length = self.spectrogram_args.HOP_LENGTH,
            n_mels = self.spectrogram_args.N_MELS,
            window_fn = torch.hamming_window,
            )
        self.spectrogram_t = lambda n_frames: (np.arange(n_frames)*self.spectrogram_args.HOP_LENGTH)/self.sr
        self.spectrogram_f = get_torch_mel_frequencies(f_max=self.spectrogram_args.F_MAX, f_min=self.spectrogram_args.F_MIN, n_mels=self.spectrogram_args.N_MELS).numpy()[1:-1] # Using defaults

    def __getitem__(self, idx):
        """ Map int idx to dict of torch tensors """
        start = idx * self.clip_hop

        audio, file_sr = librosa.load(self.audio_fp, sr=None, offset=start, duration=self.clip_duration, mono=True)
        audio = torch.from_numpy(audio)

        audio = audio-torch.mean(audio)
        if file_sr != self.sr:
          audio = torchaudio.functional.resample(audio, file_sr, self.sr)
          audio = crop_and_pad(audio, self.sr, self.clip_duration)

        record = {"sound_name": self.audio_fp, "start_time": start}
        mel_spectrogram = self.make_mel_spectrogram(audio) # size: (channel if any, n_mels, time)
        mel_spectrogram_dB = DetectronDataset.power_to_dB(self, mel_spectrogram)
        record["image"] = DetectronDataset.spectrogram_to_image(self, mel_spectrogram_dB)
        record["height"] = mel_spectrogram.shape[0] #f
        record["width"] = mel_spectrogram.shape[1] #t

        return record

class SoundEventTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        #TODO: do we want multi-gpu training?
        train_info_df = pd.read_csv(cfg.SOUND_EVENT.train_info_fp)
        for epoch_idx in range(cfg.SOUND_EVENT.n_epochs):
            print(f"Starting epoch {epoch_idx}")
            dataset = DetectronDataset(cfg, train_info_df, True, cfg.SOUND_EVENT, random_seed_shift = epoch_idx)
            effective_batch_size = cfg.SOUND_EVENT.batch_size*2 if cfg.SOUND_EVENT.mixup else cfg.SOUND_EVENT.batch_size
            data_loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, num_workers=cfg.SOUND_EVENT.num_workers, collate_fn=create_collate_fn(cfg, dataset))
            yield from data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """ NOTE - see params.py: dataset_name should be cfg.SOUND_EVENT.val_info_fp, but could add additional info_csvs into the list there if desired. """
        val_info_df = pd.read_csv(dataset_name) #Could also use cfg.SOUND_EVENT.val_info_fp directly
        dataset = DetectronDataset(cfg, val_info_df, False, cfg.SOUND_EVENT)
        return DataLoader(dataset, batch_size=None, collate_fn=list_collate)

    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name):
    # TODO (high priority): create this

def list_collate(list_of_datapoints):
    """ Without this, Dataloader will attempt to batch the record dict
        However, detectron accepts list[dict]
        See: https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format
        therefore instead of attempting to collate into dict[Tensor], this simply returns the list[dict]
    """
    return list_of_datapoints

def create_collate_fn(cfg, dataset):
    """ Create a collate function with empty annotation filter and mixup as options"""
    def collate_fn(D):
        """ Maintain the list[dict] format of the output of Dataset
        Without this collate function, Dataloader will attempt to batch the list of record dicts from Dataset
        However, detectron accepts list[dict]
        See: https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format
        Therefore instead of attempting to collate into dict[Tensor], this function simply returns the list[dict]
        """
        if cfg.SOUND_EVENT.mixup:
            #Add each sound to another, and combine the annotations
            if len(D) > 1:
                rD = D[::-1]
                new_records = []
                for idx in range(len(D)//2):
                    mixed_audio = rD[idx]["audio"] + D[idx]["audio"]
                    mel_spectrogram = dataset.make_mel_spectrogram(mixed_audio)
                    mel_spectrogram_dB = dataset.power_to_dB(mel_spectrogram)
                    image = dataset.spectrogram_to_image(mel_spectrogram_dB)
                    instances = Instances(mel_spectrogram.shape)
                    instances.gt_boxes = Boxes.cat([rD[idx]["instances"].gt_boxes, D[idx]["instances"].gt_boxes])
                    instances.gt_classes = torch.cat([rD[idx]["instances"].gt_classes,D[idx]["instances"].gt_classes])
                    new_records.append({
                        "image": image,
                        "height": D[idx]["height"],
                        "width": D[idx]["width"],
                        "instances": instances
                    })
                D = new_records
        return D
    return collate_fn

def collect_dataset_statistics(cfg, n_train_samples=2000, box_search_multiplier=3, use_box_statistics=False):
    """Determine data-related config params (regarding spectrogram and boxes), adapted to training set """

    #Get the minimum of the power spectrogram to determine the spectrogram reference value
    train_info_df = pd.read_csv(cfg.SOUND_EVENT.train_info_fp)
    dataset = DetectronDataset(cfg, train_info_df, True, cfg.SOUND_EVENT, collect_statistics=True)
    min_val = []
    n_train_samples = min(n_train_samples, len(dataset))
    print(f"~~~ Setting cfg values based on {n_train_samples} samples from train set.")

    for example_idx in range(n_train_samples):
        p = dataset[example_idx]["power"]
        min_val.append(p[p>0].min().item())
    ref = min(min_val) * 0.1
    print("Using spectrogram ref=",  ref)
    cfg.SPECTROGRAM.REF = ref

    #Having defined the spectrogram reference value, obtain the pixels
    dataset = DetectronDataset(cfg, train_info_df, True, cfg.SOUND_EVENT, collect_statistics=False)
    images = []; box_info = []; random_idxs = np.random.permutation(len(dataset))[:n_train_samples]
    for example_idx in random_idxs:
        r = dataset[example_idx]
        images.append(r["image"][0, :, :])
    #Define the pixel mean and standard deviation based on the training samples
    pixel_mean = torch.mean(torch.stack(images)).item()
    pixel_std = torch.mean(torch.std(torch.stack(images,dim=0),dim=[1,2])).item()
    print(f"Pixel mean: {pixel_mean}, Pixel std: {pixel_std}")
    print(f"(Does not affect cfg) Pixel max: {torch.stack(images).max().item()}, Pixel min: {torch.stack(images).min().item()}")
    cfg.MODEL.PIXEL_MEAN[0] = pixel_mean
    cfg.MODEL.PIXEL_STD[0] = pixel_std

    #Finally collect boxes to compute box statistics
    omit_empty_cfg = cfg.clone() #Do not change original config
    omit_empty_cfg.SOUND_EVENT.omit_empty_clip_prob = 1.
    dataset = DetectronDataset(omit_empty_cfg, train_info_df, True, omit_empty_cfg.SOUND_EVENT, collect_statistics=False)
    box_info = []; example_idx = 0; random_idxs = np.random.permutation(len(dataset))
    if use_box_statistics:
        while (len(box_info) < n_train_samples) and (example_idx < len(dataset)):
            r = dataset[random_idxs[example_idx]]
            for box in r["instances"].gt_boxes:
                width = (box[2] - box[0]).item()
                height = (box[3] - box[1]).item()
                box_size = np.sqrt(width*height)
                aspect_ratio = height/width
                box_info.append((box_size, aspect_ratio))
            example_idx += 1
            if example_idx > box_search_multiplier*n_train_samples:
                print(f"In {box_search_multiplier*n_train_samples} datapoints, could only find {len(box_info)} boxes < {n_train_samples}")
                break

    #Determine and define a set of anchor parameters from the box statistics
    if len(box_info) > 0 and use_box_statistics:
        print("Total boxes found: ", len(box_info))
        box_sizes = np.array(box_info) #n_boxes, 2
        # Compute quantiles of box stats
        qs = [0.05, 0.25, 0.5, 0.75, 0.95] if len(cfg.MODEL.RPN.IN_FEATURES) == 1 else np.linspace(0.05, 0.95, len(cfg.MODEL.RPN.IN_FEATURES))
        box_size_quantiles = np.round(np.quantile(box_sizes[:,0], qs)).astype(int)
        aspect_ratio_quantiles = np.round(10**np.quantile(np.log10(box_sizes[:,1]), [0.125,0.5,0.875]),decimals=3)
        print(f"Box size {qs} quantiles: ", box_size_quantiles)
        print(f"Aspect ratio {qs} quantiles: ", aspect_ratio_quantiles)
        if (len(np.unique(box_size_quantiles)) == 1) or (len(np.unique(aspect_ratio_quantiles)) == 1):
            warnings.warn("Single size or aspect ratio may cause issues for loss_rpn_loc.")
        # Set config to box stats
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[float(l) for l in list(np.unique(aspect_ratio_quantiles))]]
        # import pdb; pdb.set_trace()
        if len(cfg.MODEL.RPN.IN_FEATURES) == 1:
            box_size_quantiles = np.unique(box_size_quantiles)
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[int(l) for l in list(box_size_quantiles)]]
        else:
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[int(l)] for l in list(box_size_quantiles)]
        # Save a picture of the box statistic distribution for later inspection
        df = pd.DataFrame({"Size": box_sizes[:,0], "Log10(Aspect ratio)": np.log10(box_sizes[:,1])})
        sns.jointplot(x="Size", y="Log10(Aspect ratio)", data=df)
        plt.savefig(cfg.SOUND_EVENT.experiment_dir + "/box_stats.png")
        plt.close()
        print(f"Using Box Sizes: {cfg.MODEL.ANCHOR_GENERATOR.SIZES}, Aspect Ratios: {cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS}")
    else:
        print(f"No boxes, using cfg instead. Sizes: {cfg.MODEL.ANCHOR_GENERATOR.SIZES}, Aspect Ratios: {cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS}")

    print("~~~~")
    return cfg
