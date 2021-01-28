"""
Created by José Vicente Egas López
on 2021. 01. 27. 11 57

File intended for frame-level feature extraction such as fbanks, mfccs, spectrograms, etc.
Note: Functionalities are going to be implemented as needed
"""

import torchaudio


def compute_fbanks_torch(file_path, **params):
    """ :return fbank tensor containing the fbanks (Kaldi would output an identical one)"""
    waveform, sr = torchaudio.load(file_path)
    fbank = torchaudio.compliance.kaldi.fbank(waveform=waveform, **params)
    return fbank




