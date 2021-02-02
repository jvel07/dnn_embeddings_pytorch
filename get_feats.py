"""
Created by José Vicente Egas López
on 2021. 01. 27. 11 57

File intended for frame-level feature extraction such as fbanks, mfccs, spectrograms, etc.
Note: Functionalities are going to be implemented as needed
"""
import os

import torchaudio
import numpy as np


def execute_extraction_function(feat_type, waveform=None, **params):
    """Switcher to select a specific feature extraction function
    Args:
        feat_type (string): Type of the frame-level feature to extract from the utterances.
                                Choose from: 'mfcc', 'fbanks', 'spec'.
        waveform (Tensor): Tensor object containing the waveform.
        **params: Parameters belonging to the corresponding feature extraction function.
    """
    switcher = {
        'mfcc': lambda: torchaudio.compliance.kaldi.mfcc(waveform=waveform, **params),
        'fbanks': lambda: torchaudio.compliance.kaldi.fbank(waveform=waveform, **params),
        'spec': lambda: torchaudio.compliance.kaldi.spectrogram(waveform=waveform, **params),
        'melspecT': lambda: torchaudio.transforms.MelSpectrogram(**params)(waveform),
    }
    return switcher.get(feat_type, lambda: "Error, feature extraction function {} not supported!")()


class FLevelFeatsTorch(object):

    def __init__(self, save=None, out_dir=None, feat_type='fbanks', **params):
        """
        Compute fbanks of an audio signal using Kaldi in PyTorch.
        Args:
            save (boolean): Boolean, if the features have to be saved to disk set it to True;
                            False otherwise.
            out_dir (string): Destination dir of the features, use when 'save=True'.
            feat_type (string): Type of the frame-level feature to extract from the utterances.
                                Choose from: 'mfcc', 'fbanks', 'melspec'. Default is: 'fbanks'.
            **params (dictionary): Params of the fbanks.
        """
        self.feat_type = feat_type
        self.params = params
        self.save = save
        self.out_dir = out_dir

    def __call__(self, sample, wav_file):
        waveform, label, wav_file = sample['wave'], sample['label'], sample['wav_file']
        save = self.save
        out_dir = self.out_dir + '/{}/'.format(self.feat_type)
        params = self.params
        # Compute features
        feat = execute_extraction_function(feat_type=self.feat_type, waveform=waveform, **params)
        # Save features
        if save:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            np.save(out_dir + '/{0}_{1}'.format(self.feat_type, wav_file), feat.numpy())
        feature = {'fbanks': feat, 'label': label}

        return feature


class MelSpecTorch(object):

    def __init__(self, save=None, out_dir=None, **params):
        """
        Compute mel-spectrograms of an audio signal using torchaudio.transforms
        Args:
            save (boolean): Boolean, if the features have to be saved to disk set it to True;
                            False otherwise.
            out_dir (string): Destination dir of the features, use when 'save=True'.
            **params (dictionary): Params of the fbanks.
        """
        self.params = params
        self.save = save
        self.out_dir = out_dir

    def __call__(self, sample, wav_file):
        waveform, label, wav_file = sample['wave'], sample['label'], sample['wav_file']
        save = self.save
        out_dir = self.out_dir + '/mfcc/'
        params = self.params
        # Compute features
        fbank = torchaudio.compliance.kaldi.fbank(waveform=waveform, **params)
        # Save features
        if save:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            np.save(out_dir + '/mfcc_{}'.format(wav_file), fbank.numpy())
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
