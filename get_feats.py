"""
Created by José Vicente Egas López
on 2021. 01. 27. 11 57

File intended for frame-level feature extraction such as fbanks, mfccs, spectrograms, etc.
Note: Functionalities are going to be implemented as needed
"""

import torchaudio


class FbanksKaltorch(object):

    def __init__(self, **params):
        """
        Compute fbanks of an audio signal using Kaldi in PyTorch.
        Args:
            **params (dictionary): Params of the fbanks.
        """
        self.params = params

    def __call__(self, sample):
        waveform, label = sample['wave'], sample['label']
        params = self.params
        fbank = torchaudio.compliance.kaldi.fbank(waveform=waveform, **params)
        feature = {'fbanks': fbank, 'label': label}

        return feature


class MFCCTorch(object):

    def __init__(self, **params):
        """
        Compute MFCCs of an audio signal using Kaldi in PyTorch.
        Args:
            **params (dictionary): Params of the fbanks.
        """
        self.params = params

    def __call__(self, sample):
        waveform, label = sample['wave'], sample['label']
        params = self.params
        fbank = torchaudio.compliance.kaldi.mfcc(waveform=waveform, **params)
        feature = {'mfcc': fbank, 'label': label}

        return feature
