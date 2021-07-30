"""
Created by José Vicente Egas López
on 2021. 01. 28. 13 40

Class of type torch.utils.data.Dataset for loading the datset as per PyTorch
"""
import os
from abc import ABC

import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import utils


class CustomAudioDataset(Dataset):

    def __init__(self, file_labels, audio_dir, max_length_sec, online=True, feats_fir=None, calc_flevel=None):
        """ Class to load a custom Dataset. Can be used as an input for the DataLoader.
        This class is intended for loading audio data.
        Args:
            file_labels (string): Path to the csv file with the labels.
            audio_dir (string): Path to the WAV utterances.
            online (boolean, optional): if True, features are computed on the fly.
                                        if False, features are loaded from disk. Default: True
            feats_fir (string, optional): The directory containing the files of the features (use only if
                                        'online'=False). Default: None.
            calc_flevel (callable, optional): Optional calculation to be applied on a sample. E.g. compute fbanks
                                            or MFCCs of the audio signals. Use when online=True.
            max_length_sec (int): Maximum length in seconds to keep from the utterances.
        :return dictionary {
        """
        name_set = os.path.basename(feats_fir)
        self.labels = utils.load_labels(file_labels, name_set)
        self.list_wavs = utils.get_files_abspaths(path=audio_dir + name_set, file_type='.wav')
        self.name_set = name_set
        self.calc_flevel = calc_flevel
        self.online = online
        self.max_length_sec = max_length_sec
        if not online:
            self.list_feature_files = utils.get_files_abspaths(path=feats_fir, file_type='.npy')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        wav_file = self.list_wavs[idx]
        class_id = self.labels[idx]
        wav_name = os.path.basename(os.path.splitext(wav_file)[0])  # getting the basename of the wav
        name_set = self.name_set
        max_length_sec = self.max_length_sec

        if self.online:
            # self.feats_info = None
            # waveform = utils.load_wav(wav_file, sr=16000, min_dur_sec=4)
            waveform = utils.load_wav_torch(wav_file, max_length_in_seconds=max_length_sec, pad_and_truncate=True)
            sample = {
                'wave': waveform, 'label': class_id#, 'wav_file': wav_name
            }
            if self.calc_flevel:
                sample = self.calc_flevel(sample, wav_name, name_set)

        else:
            self.online = False
            feat_file_path = self.list_feature_files[idx]
            features = np.load(feat_file_path)
            sample = {
                'feature': features, 'label': class_id
            }

        return sample


class DementiaDataset(Dataset, ABC):
    """ Class to load a custom Dataset. Can be used as an input for the DataLoader.
    This class loads the transcriptions of the SZTE-DEMENTIA corpus.
    Args:
        transcriptions_dir (string): Path to the txt files containing the transcriptions.
        labels_dir (string): Path to the csv file with the labels.
        tokenizer (Tokenizer Class from HF): Tokenizer instantiated object.
        max_len (int): Maximum length in number of tokens (words).
        tokens_to_exclude (List): List of the tokens to be removed from the transcription. E.g.: "[SIL]"
        calc_embeddings (callable, optional): Optional. Compute embeddings from the transcriptions using trasnformer models.
    :return dictionary {
    """
    def __init__(self, transcriptions_dir, labels_dir, tokenizer, max_len, tokens_to_exclude, calc_embeddings=None):
        self.transcriptions_dir = transcriptions_dir
        self.list_trans_files = utils.load_just_75(labels_path=labels_dir, transcriptions_path=transcriptions_dir)
        self.labels, ids = utils.load_labels_alone(labels_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tokens_to_exclude = tokens_to_exclude
        self.calc_embeddings = calc_embeddings

    def __len__(self):
        return len(self.list_trans_files)

    def __getitem__(self, item):
        transcription_file = self.list_trans_files[item]
        transcription_text = utils.load_and_process_trans(transcription_file, tokens_to_exclude=self.tokens_to_exclude,
                                                          lower_case=True)
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            transcription_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        sample = {
            'transcription': transcription_text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'transcription_id': os.path.basename(transcription_file)
        }

        if self.calc_embeddings:
            sample = self.calc_embeddings(sample, os.path.basename(transcription_file))

        return sample

