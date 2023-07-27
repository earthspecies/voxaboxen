import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
from torch.utils.data import DataLoader

from detectron2.engine import DefaultTrainer
from detectron2.structures import Instances, Boxes, BitMasks
import detectron2.utils.comm as comm
from intervaltree import IntervalTree

from source.data.data import DetectionDataset, crop_and_pad

def spectrogram_to_image(spectrogram, image_options):
    """Add channels to basic spectrogram to make up for differences with (visual) images"""
    img = spectrogram[None, :, :] #Should be channel=1, f, t

    mask = (img > image_options.FLOOR_THRESHOLD).to(img)
    img = (img - image_options.FLOOR_THRESHOLD) * mask
    img = (img / image_options.CEIL_THRESHOLD)*255.0

    boundaries = torch.ones(img.shape, dtype=img.dtype).to(img)
    full_img = torch.cat((img, mask, boundaries), dim=0) #channels=3, freqs, time

    return full_img.to(torch.float32)

def get_torch_mel_frequencies(f_max, n_mels, f_min=0.0, mel_scale="htk"):
    m_min = torchaudio.functional.functional._hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = torchaudio.functional.functional._hz_to_mel(f_max, mel_scale=mel_scale)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = torchaudio.functional.functional._mel_to_hz(m_pts, mel_scale=mel_scale)
    return f_pts

class DetectronDataset(DetectionDataset):
    def __init__(self, detectron_cfg, info_df, train, args, random_seed_shift = 0):
        super().__init__(info_df, train, args, random_seed_shift)
        self.spectrogram_args = detectron_cfg.SPECTROGRAM
        f_max = float(self.sr // 2)
        self.make_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.sr,
            f_max = f_max,
            n_fft = self.spectrogram_args.N_FFT,
            win_length = self.spectrogram_args.WIN_LENGTH,
            hop_length = self.spectrogram_args.HOP_LENGTH,
            n_mels = self.spectrogram_args.N_MELS,
            )
        self.spectrogram_t = lambda n_frames: (np.arange(n_frames)*self.spectrogram_args.HOP_LENGTH)/self.sr
        self.spectrogram_f = get_torch_mel_frequencies(f_max=f_max, n_mels=self.spectrogram_args.N_MELS).numpy()[1:-1] # Using defaults 

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
        index_intervals = [(aa(iv[0], t), aa(iv[2], f), aa(iv[1], t), aa(iv[3], f)) for iv in intervals]
        return index_intervals

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
        mel_spectrogram = self.make_mel_spectrogram(audio) # size: (channel, n_mels, time)
        mel_spectrogram_dB = torchaudio.functional.amplitude_to_DB(mel_spectrogram[None, None, :, :], multiplier=10., amin=self.spectrogram_args.REF, db_multiplier=np.log10(self.spectrogram_args.REF)).squeeze()
        record["image"] = torch.as_tensor(spectrogram_to_image(mel_spectrogram_dB, self.spectrogram_args))
        record["height"] = mel_spectrogram.shape[0] #f
        record["width"] = mel_spectrogram.shape[1] #t

        # -- Create annotations ---
        record["instances"] = Instances(mel_spectrogram.shape)
        record["instances"].gt_classes = torch.LongTensor(np.array([pi[-1] for pi in pos_intervals]))
        #record["instances"].gt_ibm = BitMasks(masks.permute(0,2,1).contiguous())
        boxes = self.convert_intervals_to_boxes(pos_intervals, mel_spectrogram.shape[1])
        record["instances"].gt_boxes = Boxes(boxes)

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
            data_loader = DataLoader(dataset, batch_size=cfg.SOUND_EVENT.batch_size, num_workers=cfg.SOUND_EVENT.num_workers, collate_fn=list_collate)
            yield from data_loader

    @classmethod
    def build_test_loader(cls, cfg):
        test_info_df = pd.read_csv(cfg.SOUND_EVENT.test_info_fp)
        dataset = DetectronDataset(cfg, test_info_df, False, cfg.SOUND_EVENT)
        return DataLoader(dataset, batch_size=None, collate_fn=list_collate)


def list_collate(L):
    """ Without this, Dataloader will attempt to batch the record dict
        However, detectron accepts list[dict] 
        See: https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format
        therefore instead of attempting to collate into dict[Tensor], this simply returns the list[dict]
        TODO: write collate that does mixup?
    """
    return L