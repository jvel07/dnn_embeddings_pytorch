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
    return switcher.get(feat_type, lambda: "Error, feature extraction function {} not supported!".format(feat_type))()


class FLevelFeatsTorch(object):

    def __init__(self, save=None, out_dir=None, feat_type='fbanks', deltas=None, **params):
        """
        Compute frame-level features of an audio signal using Kaldi-PyTorch and PyTorch on the fly
        and OPTIONALLY save them. Note: This class is intended to be used at training time in the DataLoader.
        Args:
            save (boolean, optional): Boolean, if the features have to be saved to disk set it to True;
                            False otherwise. Default: None.
            out_dir (string, optional): Destination dir of the features, use when 'save=True'. Default: None.
            feat_type (string): Type of the frame-level feature to extract from the utterances.
                                Choose from: 'mfcc', 'fbanks', 'melspec'. Default is: 'fbanks'.
            deltas (int, optional): Compute delta coefficients of a tensor. '1' for first order derivative, '2' for second order.
                                     None for not using deltas. Default: None.
            **params (dictionary): Params of the fbanks.
        """
        self.deltas = deltas
        self.feat_type = feat_type
        self.params = params
        self.save = save
        self.out_dir = out_dir

    def __call__(self, sample, wav_file, name_set):
        waveform, label = sample['wave'], sample['label']
        save = self.save
        params = self.params
        deltas = self.deltas
        out_dir = self.out_dir

        # Compute without derivatives
        if deltas is None:
            # Compute features
            feat = execute_extraction_function(feat_type=self.feat_type, waveform=waveform, **params)
            # Save features if asked for
            out_dir = out_dir + '/{0}/{1}/'.format(self.feat_type, name_set)
            if save:
                utils.save_features(out_dir, self.feat_type, wav_file, feat)
                utils.copy_conf(out_dir, self.feat_type)
            feature = {'feature': feat, 'label': label}
            return feature

        # Compute derivatives if asked for
        if deltas == 1:
            # Compute features
            feat = execute_extraction_function(feat_type=self.feat_type, waveform=waveform, **params)
            delta1 = torchaudio.functional.compute_deltas(feat)  # compute 1st order
            feat = torch.cat((feat, delta1), 1)
            # Save features if asked for
            out_dir = out_dir + '/{0}/{1}/'.format(self.feat_type, name_set)
            if save:
                utils.save_features(out_dir, self.feat_type, '{0}_{1}del'.format(wav_file, deltas), feat)
                utils.copy_conf(out_dir, self.feat_type)
            feature = {'feature': feat, 'label': label}
            return feature
        if deltas == 2:
            # Compute features
            feat = execute_extraction_function(feat_type=self.feat_type, waveform=waveform, **params)
            delta1 = torchaudio.functional.compute_deltas(feat)  # compute 1st order
            delta2 = torchaudio.functional.compute_deltas(delta1)
            feat = torch.cat((feat, delta1, delta2), 1)
            # Save features if asked for
            out_dir = out_dir + '/{0}/{1}/'.format(self.feat_type, name_set)
            if save:
                utils.save_features(out_dir, self.feat_type, '{0}_{1}del'.format(wav_file, deltas), feat)
                utils.copy_conf(out_dir, self.feat_type)
            feature = {'feature': feat, 'label': label}
            return feature


def compute_feats_offline(source_path, name_set, out_dir, feat_type, deltas=None):
    """Function to calculate the frame-level features and save them into files.
    The function saves one file (containing features) per utterance
    Args:
        source_path (string): Path to the wavs.
        name_set (string, optional): Name of the subfolder. E.g.: train, dev, test.
        out_dir (string): Type of the frame-level feature to extract from the utterances.
                          Choose from: 'mfcc', 'fbanks', 'melspec'. Default is: 'fbanks'.
        feat_type (string): Type of the frame-level feature to extract from the utterances.
                            Choose from: 'mfcc', 'fbanks', 'melspec'. Default is: 'fbanks'.
        deltas (int, optional): Compute delta coefficients of a tensor. '1' for first order derivative, '2' for second order.
                                None for not using deltas. Default: None.
    """
    list_wavs = utils.get_files_abspaths(path=source_path + name_set, file_type='.wav')
    # frame-level feats params/config from the config file
    params = utils.read_conf_file(file_name='{}.ini'.format(feat_type), conf_section='DEFAULT')

    for wav_file in list_wavs:
        # Load wav
        waveform = utils.load_wav_torch(wav_file, max_length_in_seconds=5, pad_and_truncate=True)

        # Compute without derivatives
        if deltas is None:
            # Compute features
            feat = execute_extraction_function(feat_type=feat_type, waveform=waveform, **params)
            out_dir = out_dir + '/{0}/{1}/'.format(feat_type, name_set)
            utils.save_features(out_dir, feat_type, wav_file, feat)
            utils.copy_conf(out_dir, feat_type)

        # Compute derivatives if asked for
        if deltas == 1:
            # Compute features
            feat = execute_extraction_function(feat_type=feat_type, waveform=waveform, **params)
            delta1 = torchaudio.functional.compute_deltas(feat)  # compute 1st order
            feat = torch.cat((feat, delta1), 1)
            out_dir = out_dir + '/{0}/{1}/'.format(feat_type, name_set)
            utils.save_features(out_dir, feat_type, wav_file, feat)
            utils.copy_conf(out_dir, feat_type)

        if deltas == 2:
            # Compute features
            feat = execute_extraction_function(feat_type=feat_type, waveform=waveform, **params)
            delta1 = torchaudio.functional.compute_deltas(feat)  # compute 1st order
            delta2 = torchaudio.functional.compute_deltas(delta1)
            feat = torch.cat((feat, delta1, delta2), 1)
            out_dir = out_dir + '/{0}/{1}/'.format(feat_type, name_set)
            utils.save_features(out_dir, feat_type, wav_file, feat)
            utils.copy_conf(out_dir, feat_type)
