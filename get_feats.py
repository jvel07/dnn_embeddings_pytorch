"""
Created by José Vicente Egas López
on 2021. 01. 27. 11 57

File intended for frame-level feature extraction such as fbanks, mfccs, spectrograms, etc.
Note: Functionalities are going to be implemented as needed
"""
import os

import torch
import torchaudio
import numpy as np

import utils


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

    def __init__(self, save=None, out_dir=None, feat_type='fbanks', deltas=None, **params):
        """
        Compute frame-level features of an audio signal using Kaldi-PyTorch and PyTorch.
        Args:
            save (boolean): Boolean, if the features have to be saved to disk set it to True;
                            False otherwise.
            out_dir (string): Destination dir of the features, use when 'save=True'.
            feat_type (string): Type of the frame-level feature to extract from the utterances.
                                Choose from: 'mfcc', 'fbanks', 'melspec'. Default is: 'fbanks'.
            deltas (int): Compute delta coefficients of a tensor. '1' for first order derivative, '2' for second order.
                          None for not using deltas. Default: None.
            **params (dictionary): Params of the fbanks.
        """
        self.deltas = deltas
        self.feat_type = feat_type
        self.params = params
        self.save = save
        self.out_dir = out_dir

    def __call__(self, sample, wav_file):
        waveform, label, wav_file = sample['wave'], sample['label'], sample['wav_file']
        save = self.save
        out_dir = self.out_dir + '/{}/'.format(self.feat_type)
        params = self.params
        deltas = self.deltas

        # Compute without derivatives
        if deltas is None:
            # Compute features
            feat = execute_extraction_function(feat_type=self.feat_type, waveform=waveform, **params)
            # Save features if asked for
            if save:
                utils.save_features(out_dir, self.feat_type, wav_file, feat)
            feature = {'feature': feat, 'label': label}
            return feature

        # Compute derivatives if asked for
        if deltas == 1:
            # Compute features
            feat = execute_extraction_function(feat_type=self.feat_type, waveform=waveform, **params)
            delta1 = torchaudio.functional.compute_deltas(feat)  # compute 1st order
            feat = torch.cat((feat, delta1), 1)
            feature = {'feature': feat, 'label': label}
            # Save features if asked for
            if save:
                utils.save_features(out_dir, self.feat_type, wav_file, feat)
            return feature
        if deltas == 2:
            # Compute features
            feat = execute_extraction_function(feat_type=self.feat_type, waveform=waveform, **params)
            delta1 = torchaudio.functional.compute_deltas(feat)  # compute 1st order
            delta2 = torchaudio.functional.compute_deltas(delta1)
            feat = torch.cat((feat, delta1, delta2), 1)
            feature = {'feature': feat, 'label': label}
            # Save features if asked for
            if save:
                utils.save_features(out_dir, self.feat_type, wav_file, feat)
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
