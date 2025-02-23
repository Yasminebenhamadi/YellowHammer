#!/usr/bin/env python
# coding: utf-8

import os
import librosa
import numpy as np
from scipy.signal import butter, filtfilt


# Code for the different preprocessing

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Code from Masters students
def process_audio_band(file_path, band_ranges=None, normalize=True):
    if band_ranges is None:
        band_ranges = [(3500, 5500), (5000, 7000), (7000, 9000)]

    # Load audio file
    audio_signal, sample_rate = librosa.load(file_path)

    # Extract features for each band
    features = []
    for lowcut, highcut in band_ranges:
        # Apply bandpass filtering with downsampling
        filtered_signal = bandpass_filter(audio_signal, lowcut, highcut, sample_rate)

        envelope = librosa.feature.rms(y=filtered_signal, frame_length=1048, hop_length=128)[0]

        # Normalize if required
        if normalize:
            envelope = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope))
        features.append(envelope)

    # Concatenate envelopes from all bands
    return np.concatenate(features)

def load_process_bands(folder, sample_files, band_ranges=None, normalize=True):
    if band_ranges is None:
        band_ranges = [(3500, 5500), (5000, 7000), (7000, 9000)]

    features = []

    for file_name in sample_files:
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder, file_name)
            # Extract features
            feature = process_audio_band(file_path, band_ranges=band_ranges, normalize=normalize)
            features.append(feature)

    # Ensure all features have the same length
    max_length = max(len(f) for f in features)
    features = [np.pad(f, (0, max_length - len(f))) for f in features]
    return np.array(features)

# Code for other architectures
def load_process_avg(folder, sample_files, band_ranges):
    X = []
    for sample_file in sample_files:
        audio, samplerate = librosa.load(folder+sample_file)
        sub_band_energies = []
        for lowcut, highcut in band_ranges:
            filtered_signal = bandpass_filter(audio, lowcut, highcut, samplerate, order=5)
            energy = librosa.feature.rms(y=filtered_signal, frame_length=filtered_signal.shape[0], hop_length=1, center=False)[0,0]
            sub_band_energies.append(energy)
        X.append(sub_band_energies)
    X = np.array(X)
    return X

def load_process_envelope(folder, sample_files, f_min, f_max, frame, hop, plot=False):
    X = []
    for sample_file in sample_files:
        audio, samplerate = librosa.load(folder+sample_file)
        filtered_audio = bandpass_filter(audio, f_min, f_max, samplerate)
        sub_amplitude_envelope = librosa.feature.rms(y=filtered_audio, frame_length=frame, hop_length=hop)[0]
        X.append(sub_amplitude_envelope)
    X = np.array(X)
    return X

def load_process_mel(folder, sample_files, nb_mels, f_min, f_max, hop, nfft, plot=False):
    X = []
    for sample_file in sample_files:
        audio, samplerate = librosa.load(folder+sample_file)#, sr=None)
        mel_s = librosa.feature.melspectrogram(y=audio, sr=samplerate, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop, n_mels = nb_mels)
        X.append(mel_s)
    X = np.array(X)
    return X