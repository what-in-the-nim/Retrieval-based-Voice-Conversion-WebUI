import os
import sys
import logging
import traceback

import numpy as np
import torch
from scipy.io.wavfile import read
from torch.utils.data import Dataset

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

# Reusable banks
mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    """Convert waveform into Linear-frequency Linear-amplitude spectrogram.

    Args:
        y             :: (B, T) - Audio waveforms
        n_fft
        hop_size
        win_size
        center
    Returns:
        :: (B, Freq, Frame) - Linear-frequency Linear-amplitude spectrogram
    """

    # Window - Cache if needed
    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # Padding
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # Complex Spectrogram :: (B, T) -> (B, Freq, Frame, RealComplex=2)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # Linear-frequency Linear-amplitude spectrogram :: (B, Freq, Frame, RealComplex=2) -> (B, Freq, Frame)
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
    return spec

class TextAudioLoaderMultiNSFsid(Dataset):
    """
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text, pitch, pitchf, dv in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text, pitch, pitchf, dv])
                lengths.append(os.path.getsize(audiopath) // (3 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        file = audiopath_and_text[0]
        phone = audiopath_and_text[1]
        pitch = audiopath_and_text[2]
        pitchf = audiopath_and_text[3]
        dv = audiopath_and_text[4]

        phone, pitch, pitchf = self.get_labels(phone, pitch, pitchf)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]
        # print(123,phone.shape,pitch.shape,spec.shape)
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            # amor
            len_wav = len_min * self.hop_length

            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]

            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

        return (spec, wav, phone, pitch, pitchf, dv)

    def get_labels(self, phone, pitch, pitchf):
        phone = np.load(phone)
        phone = np.repeat(phone, 2, axis=0)
        pitch = np.load(pitch)
        pitchf = np.load(pitchf)
        n_num = min(phone.shape[0], 900)  # DistributedBucketSampler
        # print(234,phone.shape,pitch.shape)
        phone = phone[:n_num, :]
        pitch = pitch[:n_num]
        pitchf = pitchf[:n_num]
        phone = torch.FloatTensor(phone)
        pitch = torch.LongTensor(pitch)
        pitchf = torch.FloatTensor(pitchf)
        return phone, pitch, pitchf

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio
        #        audio_norm = audio / self.max_wav_value
        #        audio_norm = audio / np.abs(audio).max()

        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename)
            except:
                logger.warning("%s %s", spec_filename, traceback.format_exc())
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)