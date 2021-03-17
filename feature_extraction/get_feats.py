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
from feature_extraction.fisher_helper import *


def execute_extraction_function(feat_type, waveform=None, **params):
    """Switcher to select a specific feature extraction function
    Args:
        feat_type (string): Type of the frame-level feature to extract from the utterances.
                                Choose from: 'mfcc', 'fbanks', 'spectrogram'.
        waveform (Tensor): Tensor object containing the waveform.
        **params: Parameters belonging to the corresponding feature extraction function.
    """
    switcher = {
        'mfcc': lambda: torchaudio.compliance.kaldi.mfcc(waveform=waveform, **params),
        'fbanks': lambda: torchaudio.compliance.kaldi.fbank(waveform=waveform, **params),
        'spectrogram': lambda: torchaudio.compliance.kaldi.spectrogram(waveform=waveform, **params),
        'melspecT': lambda: torchaudio.transforms.MelSpectrogram(**params)(waveform),
        'mfccT': lambda: torchaudio.transforms.MFCC(**params)(waveform),
    }
    return switcher.get(feat_type, lambda: "Error, feature extraction function {} not supported!".format(feat_type))()


class FLevelFeatsTorch(object):

    def __init__(self, save=True, out_dir=None, feat_type='fbanks', deltas=None, config_file=None):
        """
        Compute frame-level features of an audio signal using Kaldi-PyTorch and PyTorch on the fly
        and OPTIONALLY save them. Note: This class is intended to be used at training time in the DataLoader.
        Args:
            save (boolean, optional): Boolean, if the features have to be saved to disk set it to True;
                                      False otherwise. Default: True.
            out_dir (string, optional): Destination dir of the features, use when 'save=True'. Default: None.
            feat_type (string): Type of the frame-level feature to extract from the utterances.
                                Choose from: 'mfcc', 'fbanks', 'melspec', 'spectrogram'. Default is: 'fbanks'.
            deltas (int, optional): Compute delta coefficients of a tensor. '1' for first order derivative, '2' for second order.
                                     "0" or None for not using deltas. Default: None.
            config_file (string): Path to the config (ini) file.
        """
        self.deltas = deltas
        self.feat_type = feat_type
        self.config_file = config_file
        self.save = save
        self.out_dir = out_dir

    def __call__(self, sample, wav_file, name_set):
        waveform, label = sample['wave'], sample['label']
        save = self.save
        config_file = self.config_file
        deltas = self.deltas
        out_dir = self.out_dir

        # frame-level feats params/config
        params = utils.read_conf_file(file_name=config_file, conf_section='DEFAULTS')

        # Compute without derivatives
        if deltas == 0:
            # Compute features
            feat = execute_extraction_function(feat_type=self.feat_type, waveform=waveform, **params)
            # Save features if asked for
            out_dir = out_dir + '/{0}/{1}/'.format(self.feat_type, name_set)
            if save:
                utils.save_features(out_dir, self.feat_type, wav_file, feat)
                utils.copy_conf(config_file, out_dir, self.feat_type)
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
                utils.copy_conf(config_file, out_dir, self.feat_type)
            feature = {'feature': feat, 'label': label}
            return feature
        if deltas == 2:
            # Compute features
            feat = execute_extraction_function(feat_type=self.feat_type, waveform=waveform, **params)
            delta1 = torchaudio.functional.compute_deltas(feat)  # compute 1st order
            delta2 = torchaudio.functional.compute_deltas(delta1)  # compute 2nd order
            feat = torch.cat((feat, delta1, delta2), 1)
            # Save features if asked for
            out_dir = out_dir + '/{0}/{1}/'.format(self.feat_type, name_set)
            if save:
                utils.save_features(out_dir, self.feat_type, '{0}_{1}del'.format(wav_file, deltas), feat)
                utils.copy_conf(config_file, out_dir, self.feat_type)
            feature = {'feature': feat, 'label': label}
            return feature


def compute_flfeats_offline(source_path, out_dir, feat_type, deltas=None, config_file=None):
    """Function to calculate the frame-level features and save them to files.
    The function saves one file (containing features) per utterance
    Args:
        source_path (string): Path to the wavs.
        out_dir (string): Type of the frame-level feature to extract from the utterances.
                          Choose from: 'mfcc', 'fbanks', 'melspec'. Default is: 'fbanks'.
        feat_type (string): Type of the frame-level feature to extract from the utterances.
                            Choose from: 'mfcc', 'fbanks', 'melspec'. Default is: 'fbanks'.
        deltas (int, optional): Compute delta coefficients of a tensor. '1' for first order derivative,
                                '2' for second order. None for not using deltas. Default: None.
        config_file (string): Path to the configuration file (ini).
    """
    list_wavs = utils.get_files_abspaths(path=source_path, file_type='.wav')
    # frame-level feats params/config from the config file
    params = utils.read_conf_file(file_name=config_file, conf_section='DEFAULTS')

    print("Computing {} for {} utterances in {}...".format(feat_type, len(list_wavs), source_path))

    for wav_file in list_wavs:
        # Load wav
        waveform = utils.load_wav_torch(wav_file, max_length_in_seconds=5, pad_and_truncate=True)

        # Compute without derivatives
        if deltas == 0:
            # Compute features
            feat = execute_extraction_function(feat_type=feat_type, waveform=waveform, **params)
            final_dir = out_dir + '/{0}/{1}/'.format(feat_type, os.path.basename(source_path))
            utils.save_features(final_dir, feat_type, wav_file, feat)
            utils.copy_conf(config_file, final_dir, feat_type)

        # Compute derivatives if asked for
        if deltas == 1:
            # Compute features
            feat = execute_extraction_function(feat_type=feat_type, waveform=waveform, **params)
            delta1 = torchaudio.functional.compute_deltas(feat)  # compute 1st order
            feat = torch.cat((feat, delta1), 1)
            final_dir = out_dir + '/{0}/{1}/'.format(feat_type, os.path.basename(source_path))
            utils.save_features(final_dir, feat_type, wav_file, feat)
            utils.copy_conf(config_file, final_dir, feat_type)

        if deltas == 2:
            # Compute features
            feat = execute_extraction_function(feat_type=feat_type, waveform=waveform, **params)
            delta1 = torchaudio.functional.compute_deltas(feat)  # compute 1st order
            delta2 = torchaudio.functional.compute_deltas(delta1)
            feat = torch.cat((feat, delta1, delta2), 1)
            final_dir = out_dir + '/{0}/{1}/'.format(feat_type, os.path.basename(source_path))
            utils.save_features(final_dir, feat_type, wav_file, feat)
            utils.copy_conf(config_file, final_dir, feat_type)


def extract_xvecs(source_path, out_dir, net, layer_name):
    """ Function to extract the x-vector embeddings from the specified layer.
    Function based on https://github.com/manojpamk/pytorch_xvectors/
    Args:
        source_path (tensor, np array): The input features.
        layer_name (string): The name of the layer to extract the x-vectors from: 'fc1', 'fc2', 6th and 7th, resp.
        net (object): Neural Network saved model.
        out_dir (string): Output directory for the x-vectors.
    """

    list_files = utils.get_files_abspaths(source_path, '.npy')
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    eval('net.{}.register_forward_hook(get_activation(layer_name))'.format(layer_name))

    xvecs = []
    for file in list_files:
        feat = np.load(file)
        out = net(x=torch.Tensor(feat).permute(1, 0).unsqueeze(0).cuda(), eps=0)
        x_vec = np.squeeze(activation[layer_name].cpu().numpy())
        xvecs.append(x_vec)
    xvecs = np.vstack(xvecs)

    np.savetxt(out_dir + '/xvecs_512_fc1.xvecs', xvecs)
    print("x-vecs saved to {}".format(out_dir))

    return xvecs



# def train_gmm(features, )

class FisherVectors(object):

    def __init__(self, feats_dir=None, feats_gmm_dir=None, save=True, out_dir=None, config_file=None):
        """
        Compute frame-level features of an audio signal using VLFeat Note: This class is intended to be used at
        training time in the DataLoader.
        Args:
            feats_dir (string): Path to the frame-level features for Fisher vector encoding.
            feats_gmm_dir (string): Path to the frame-level features for GMM training.
            save (boolean, optional): Boolean, if the features have to be saved to disk set it to True;
                                      False otherwise. Default: True.
            out_dir (string, optional): Destination dir of the features, use when 'save=True'. Default: None.
            config_file (string): Path to the config (ini) file.
        """
        self.feats_gmm_dir = feats_gmm_dir
        self.feats_dir = feats_dir
        self.config_file = config_file
        self.save = save
        self.out_dir = out_dir

    def __call__(self, sample, wav_file, name_set):
        feature, label = sample['feature'], sample['label']
        save = self.save
        config_file = self.config_file
        out_dir = self.out_dir
        feats_gmm_dir = self.feats_gmm_dir

        # Reading params/config
        params = utils.read_conf_file(file_name=config_file, conf_section='DEFAULTS')

        # extract Fisher vectors
        # fish = extract_fishers(feature)

