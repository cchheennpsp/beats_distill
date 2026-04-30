# --------------------------------------------------------
# BEATs Knowledge Distillation - Dataset
# Lightweight audio dataset for distillation training.
# --------------------------------------------------------

import os
import csv
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T


class AudioDataset(Dataset):
    """
    A simple audio dataset for knowledge distillation training.

    Supports two modes:
    1. CSV mode: A CSV file with columns [audio_path, label_indices]
       where label_indices is a space-separated string of integer class indices.
       Example row: "/data/audio/clip1.wav", "0 137 400"
    2. Folder mode: A directory of audio files with an optional label file.

    Args:
        audio_dir: Root directory containing audio files.
        csv_path: Path to a CSV manifest file. If provided, audio_dir is used
                  as the prefix for relative paths in the CSV.
        num_classes: Total number of classes (default 527 for AudioSet).
        sample_rate: Target sample rate (default 16000).
        max_duration: Maximum audio duration in seconds. Longer clips are truncated.
                      Shorter clips are zero-padded. Default 10s.
        use_specaugment: Whether to apply SpecAugment during training (default False).
        specaugment_freq_mask_param: Frequency mask parameter for SpecAugment (default 27).
        specaugment_time_mask_param: Time mask parameter for SpecAugment (default 100).
        num_freq_masks: Number of frequency masks (default 2).
        num_time_masks: Number of time masks (default 2).
    """

    def __init__(
        self,
        audio_dir: str,
        csv_path: str = None,
        num_classes: int = 527,
        sample_rate: int = 16000,
        max_duration: float = 10.0,
        use_specaugment: bool = False,
        specaugment_freq_mask_param: int = 27,
        specaugment_time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration)
        self.use_specaugment = use_specaugment

        # Initialize SpecAugment transform if enabled
        if self.use_specaugment:
            self.specaugment = T.SpecAugment(
                n_freq_masks=num_freq_masks,
                freq_mask_param=specaugment_freq_mask_param,
                n_time_masks=num_time_masks,
                time_mask_param=specaugment_time_mask_param,
            )
        else:
            self.specaugment = None

        self.samples = []  # list of (audio_path, label_tensor)

        if csv_path is not None:
            self._load_from_csv(csv_path)
        else:
            self._load_from_folder()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No audio samples found. audio_dir={audio_dir}, csv_path={csv_path}"
            )

    def _load_from_csv(self, csv_path: str):
        """Load samples from a CSV file.

        Expected CSV format (with header):
            audio_path,label_indices
            clip1.wav,0 137 400
            clip2.wav,12

        Paths can be absolute or relative to self.audio_dir.
        """
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_path = row["audio_path"]
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(self.audio_dir, audio_path)

                # Parse multi-hot labels
                label = torch.zeros(self.num_classes)
                if "label_indices" in row and row["label_indices"].strip():
                    indices = [int(x) for x in row["label_indices"].strip().split()]
                    for idx in indices:
                        if 0 <= idx < self.num_classes:
                            label[idx] = 1.0

                self.samples.append((audio_path, label))

    def _load_from_folder(self):
        """Load all audio files from the folder (no labels — for unsupervised KD)."""
        audio_extensions = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        for fname in sorted(os.listdir(self.audio_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in audio_extensions:
                audio_path = os.path.join(self.audio_dir, fname)
                label = torch.zeros(self.num_classes)  # dummy label
                self.samples.append((audio_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]

        # Load waveform
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono and ensure 1D shape: (samples,)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        # Pad or truncate to fixed length
        if waveform.shape[0] > self.max_length:
            waveform = waveform[: self.max_length]
        elif waveform.shape[0] < self.max_length:
            pad_size = self.max_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        # Apply SpecAugment if enabled (only during training)
        if self.use_specaugment and self.specaugment is not None:
            # SpecAugment expects (channel, freq, time) format
            # Convert waveform to spectrogram-like format for augmentation
            # For BEATs, we apply SpecAugment on the waveform level by converting to fbank first
            # However, since SpecAugment works on spectrograms, we'll apply it after fbank extraction
            # For now, we return the waveform and apply SpecAugment in the training loop
            pass

        return waveform, label
