"""
Created by José Vicente Egas López
on 2021. 01. 28. 13 40

Class of type torch.utils.data.Dataset for loading the datset as per PyTorch
"""
import os

import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import utils


class CustomDataset(Dataset):

    def __init__(self, file_labels, audio_dir, name_set, online=False, feats_info=None, calc_flevel=None):
        """
        Args:
            file_labels (string): Path to the csv file with the labels.
            audio_dir (string): Path to the WAV utterances.
            name_set (string): name of the dataset (train, dev, test).
            online (boolean, optional): if True, features are computed on the fly.
                                        if False, features are loaded from disk. Default: False
            feats_info (list, optional): Optional list with TWO elements. The first is the directory containing
                                        the files of the features, the second is the type of the file (the file ext.)
                                        that contain the features. (use only if 'online'=False). Default: None
            calc_flevel (callable, optional): Optional calculation to be applied on a sample. E.g. compute fbanks
                                            or MFCCs of the audio signals. Use when online=True.
        :return dictionary {
        """
        if feats_info is None:
            feats_info = []
        else:
            self.list_feature_files = utils.get_files_abspaths(path=feats_info[0] + name_set, file_type=feats_info[1])

        self.labels = utils.load_labels(file_labels, name_set)
        self.list_wavs = utils.get_files_abspaths(path=audio_dir + name_set, file_type='.wav')
        self._set = name_set
        self.calc_flevel = calc_flevel
        self.online = online

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        wav_file = self.list_wavs[idx]
        class_id = self.labels[idx]
        wav_name = os.path.basename(os.path.splitext(wav_file)[0])

        if self.online:
            waveform = utils.load_wav(wav_file, sr=16000, min_dur_sec=4)
            sample = {
                'wave': waveform, 'label': class_id,
                'wav_file': wav_name
            }
        else:
            feat_file_path = self.list_feature_files
            features = utils.load_features_acc_file(feat_file_path)
            sample = {
                'features': features, 'label': class_id
            }
        if self.calc_flevel:
            sample = self.calc_flevel(sample, wav_name)

        return sample



