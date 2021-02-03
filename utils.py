"""
Created by José Vicente Egas López
on 2021. 01. 27. 11 57

File intended for common utils like load file paths, etc.
"""
import random

import librosa
import os
import numpy as np
import pandas as pd

# Traverse directories and pull specific type of file (".WAV", etc...)
import torch
import torchaudio


def get_files_abspaths(path, file_type=None):
    """
    Args:
        path (string): Path to the folder containing files.
        file_type (string): File format, e.g., '.wav', '.mfcc', 't', 'p', npy', etc.
    :return list of files with the absolute path
    """
    if os.path.isdir(path):
        lista = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(file_type):
                    lista.append(os.path.join(root, file))
        lista.sort()
        return lista
    else:
        print("\nERROR: Path '{}' does not exist!".format(path))
        raise FileNotFoundError


# Load wav and fix it to a specific length
def load_wav(audio_filepath, sr, min_dur_sec=5):
    audio_data, fs = librosa.load(audio_filepath, sr=16000)
    len_file = len(audio_data)

    if len_file < int(min_dur_sec * sr):
        dummy = np.zeros((1, int(min_dur_sec * sr) - len_file))
        extended_wav = np.concatenate((audio_data, dummy[0]))
    else:
        extended_wav = audio_data

    extended_wav = torch.from_numpy(extended_wav)
    return extended_wav.reshape(1, -1) # reshaping to (channel, samples) as needed in https://pytorch.org/audio/stable/compliance.kaldi.html


def load_wav_torch(audio_filepath, max_length_in_seconds, pad_and_truncate):
    audio_tensor, sample_rate = torchaudio.load(audio_filepath, normalization=True)
    max_length = sample_rate * max_length_in_seconds
    audio_size = audio_tensor.size()

    if pad_and_truncate:
        if audio_size[1] < max_length:
            difference = max_length - audio_size[1]
            padding = torch.zeros(audio_size[0], difference)
            padded_audio = torch.cat([audio_tensor, padding], 1)
            return padded_audio

        if audio_size[1] > max_length:
            random_idx = random.randint(0, audio_size[1] - max_length)
            return audio_tensor.narrow(1, random_idx, max_length)

    return audio_tensor


def load_features_acc_file(filepath):
    """
    Loads file containing features according to the extension of the file.
    Args:
        filepath (string): Path to the file containing the features.
    :return python object containing the loaded features
    """
    filepath = filepath[0]  # from list to string

    extension = os.path.splitext(filepath)[1]  # getting the basename of the file
    if extension in ['.mfcc', '.fbanks', '.spec']:  # getting the ext. and loading
        feats = np.load(filepath, allow_pickle=True)
    elif extension in ['.pt', '.pth']:
        feats = torch.load(filepath)
    elif extension in ['.npy']:
        feats = np.load(filepath)
    else:
        feats = None
        print("File format {} not supported. Try '.npy' or, '.pt' or '.pth' (PyTorch's). ".format(extension))

    return feats


def load_labels(filepath, name_set):
    """
    Loads labels from a file of containing the form, e.g.:
                                        file_name,label
                                        train_0001.wav,7
    Args:
        filepath (string): Path to the file containing the features.
        name_set (string): Set of the labels (train, dev, test)
    :return object
    """

    df = pd.read_csv(filepath)
    df_labels = df[df['file_name'].str.match(name_set)]
    labels = df_labels.label.values

    return labels


def save_features(out_dir, feat_type, wav_file, features):
    """
    Saves the features as numpy arrays to disk. Intended for get_feats.FLevelFeatsTorch().
    Args:
        out_dir (string): Output dir.
        feat_type (string): Type of the feature to be saved. E.g.: 'mfcc', 'fbanks', 'spec'
        wav_file: Name of the wav file.
        features: Torch
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.save(out_dir + '/{0}_{1}'.format(feat_type, wav_file), features.numpy())
    # print("{0} features saved to {1}".format(feat_type, out_dir))


