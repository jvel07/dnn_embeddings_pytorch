"""
Created by José Vicente Egas López
on 2021. 01. 27. 11 57

File intended for common utils like load file paths, etc.
"""
import configparser
import pathlib
import random

# import librosa
import os
import numpy as np
import pandas as pd
from shutil import copyfile
from sklearn import preprocessing

import torch
import torchaudio
from torch.nn.utils.rnn import pack_sequence


def get_files_abspaths(path, file_type=None):
    """Traverse directories and pull specific type of file (".WAV", etc...)
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
def load_wav(audio_filepath, sr, min_dur_sec):
    audio_data, fs = librosa.load(audio_filepath, sr=16000)
    len_file = len(audio_data)

    if len_file < int(min_dur_sec * sr):
        dummy = np.zeros((1, int(min_dur_sec * sr) - len_file))
        extended_wav = np.concatenate((audio_data, dummy[0]))
    else:
        extended_wav = audio_data

    extended_wav = torch.from_numpy(extended_wav)
    return extended_wav.reshape(1, -1)  # reshaping to (channel, samples) as
    # needed in https://pytorch.org/audio/stable/compliance.kaldi.html


def load_wav_torch(audio_filepath, max_length_in_seconds, pad_and_truncate):
    audio_tensor, sample_rate = torchaudio.load(audio_filepath, normalize=True)
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
    Loads features contained in a file according to the extension of it.
    Args:
        filepath (string): Path to the file containing the features.
    :return python object containing the loaded features
    """
    # filepath = filepath[0]  # from list to string

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


def read_conf_file(file_name, conf_section):
    """Read .ini files containing the parameters for computing the frame-level features.
    Args
        file_name (string): Path to the .ini file.
        conf_section (string): Name of the configuration section to read inside the .ini file.
    :return: dict with the values of the parameters.
    """
    dict_section_values = {}
    config = configparser.ConfigParser(delimiters='=', inline_comment_prefixes='#')
    config.read(file_name)
    for param in config.options(conf_section):
        value = eval(config.get(conf_section, param))
        dict_section_values[param] = value

    return dict_section_values


def load_labels(labels_dir, name_set):
    """
    Loads labels from a specific set ('name_set'), e.g.: train, from a file of containing the form, e.g.:
                                        file_name,label
                                        train_0001.wav,7
    Args:
        labels_dir (string): Path to the folder containing the labels file.
        name_set (string): Set of the labels (train, dev, test)
    :return object
    """

    # df = pd.read_csv(filepath+'/{}_orig.csv'.format(name_set), delimiter=',')
    df = pd.read_csv(labels_dir + '/labels.csv', delimiter=',')
    df['label'] = df['label'].astype('category')
    df['cat_lbl'] = df['label'].cat.codes
    df_labels = df[df['filename'].str.match(name_set)]
    labels = df_labels.cat_lbl.values
    # le = preprocessing.LabelEncoder()
    # labels = le.fit_transform(labels)
    # labels = torch.from_numpy(np.asarray(labels).astype('int64'))
    # labels_hot = torch.nn.functional.one_hot(labels)

    return labels  # , df_labels.file_name.values


def load_labels_alone(labels_file):
    """
    Loads all labels contained in a file of the form, e.g.:
                                        file_name,label
                                        train_0001.wav,7
    Args:
        labels_file (string): Path to the folder containing the labels file.
    :return object
    """

    df = pd.read_csv(labels_file, delimiter=',', dtype=str)
    df['label'] = df['label'].astype('category')
    df['cat_lbl'] = df['label'].cat.codes
    labels = df.cat_lbl.values
    ids = df.filename.values

    return labels, ids


def save_features(out_dir, feat_type, wav_file, features):
    """
    Saves the features as numpy arrays to disk. Intended for using with get_feats.FLevelFeatsTorch().
    Args:
        out_dir (string): Output dir.
        feat_type (string): Type of the feature to be saved. E.g.: 'mfcc', 'fbanks', 'spec'
        wav_file (string): Name of the wav file.
        features (Torch): Torch
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    wav_name = os.path.splitext(os.path.basename(wav_file))[0]

    file_name = '/{0}_{1}'.format(feat_type, wav_name)
    np.save(out_dir + '/' + file_name, features.numpy())


def copy_conf(orig_conf_file, out_dir, feat_type):
    conf_file_bk_path = '{0}/conf/'.format(out_dir)
    if not os.path.exists(conf_file_bk_path):
        os.makedirs(conf_file_bk_path)
    copyfile(orig_conf_file, '{0}/{1}.ini'.format(conf_file_bk_path, feat_type))


# E.g.: (4, 'mask', 'fisher') or 'ivecs'
# example: train/fisher-23mf-0del-2g-train.fisher
# loads data (files' name-format that were generated by this SW) existing in the folders "train", "dev" and "test".
# Format of the data labels required and file with the following headers:
#    'file_name     label'  Example: 'file_name     label'
#    'recording.wav label'. Example: 'train_0001.wav True', 'train_0002.wav 2 False', ...
def load_data_full(data_path, layer_name):
    list_datasets = ['train', 'dev', 'test']  # names for the datasets
    dict_data = {}
    # Load train, dev, test
    for item in list_datasets:
        # Set data directories
        file_dataset = data_path + '/{0}/xvecs_512_{1}.xvecs'.format(item, layer_name)
        # Load datasets
        dict_data['x_' + item] = np.loadtxt(file_dataset)
        # Load labels
        lbl_path = os.path.dirname(os.path.dirname(data_path))
        file_lbl_train = lbl_path + '/labels/labels_bk.csv'
        df = pd.read_csv(file_lbl_train)
        df_labels = df[df['file_name'].str.match(item)]
        dict_data['y_' + item] = df_labels.label.values
    return dict_data['x_train'], dict_data['x_dev'], dict_data['x_test'], dict_data['y_train'], dict_data['y_dev'], \
           dict_data['y_test'], file_dataset


# linear transformation for predictions (see sleepiness paper)
def linear_trans_preds(y_train, preds_dev, preds_test_orig):
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)

    mean_preds_dev = np.mean(preds_dev)
    std_preds_dev = np.std(preds_dev)

    preds_dev_new = (preds_dev - mean_preds_dev) / std_preds_dev * std_y_train + mean_y_train
    preds_dev_new = np.round(preds_dev_new)
    preds_dev_new[preds_dev_new < 1] = 1
    preds_dev_new[preds_dev_new > 9] = 9

    preds_test_new = (preds_test_orig - mean_preds_dev) / std_preds_dev * std_y_train + mean_y_train
    preds_test_new = np.round(preds_test_new)
    preds_test_new[preds_test_new < 1] = 1
    preds_test_new[preds_test_new > 9] = 9

    return preds_dev_new, preds_test_new


def linear_trans_preds_dev(y_train, preds_dev):
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)

    mean_preds_dev = np.mean(preds_dev)
    std_preds_dev = np.std(preds_dev)

    preds_dev_new = (preds_dev - mean_preds_dev) / std_preds_dev * std_y_train + mean_y_train
    preds_dev_new = np.round(preds_dev_new)
    preds_dev_new[preds_dev_new < 1] = 1
    preds_dev_new[preds_dev_new > 9] = 9

    return preds_dev_new


def linear_trans_preds_test(y_train, preds_dev, preds_test):
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)

    mean_preds_dev = np.mean(preds_dev)
    std_preds_dev = np.std(preds_dev)

    preds_test_new = (preds_test - mean_preds_dev) / std_preds_dev * std_y_train + mean_y_train
    preds_test_new = np.round(preds_test_new)
    preds_test_new[preds_test_new < 1] = 1
    preds_test_new[preds_test_new > 9] = 9

    return preds_test_new


# END of linear transformation for predictions (see sleepiness paper) ##


# utils for DEMENTIA SZTE DATASET ###

def load_just_75(labels_file, transcriptions_path):
    """
    Function that compares the transcription list and loads only the existing 75 of the labels.
    Args:
        labels_file: Path to the folder containing the 'labels.csv' file.
        transcriptions_path: Path to the folder containing the transcription files. E.g.: 'data/text/dementia94B'

    Returns:
        List of occurrences.
    """
    labs, ids = load_labels_alone(labels_file)
    list_trans_files = get_files_abspaths(path=transcriptions_path, file_type='.lab')  # ".lab" is the format provided
    occurrences = []

    for id in list_trans_files:
        for id2 in ids:
            if os.path.basename(id)[0:3] == id2:
                occurrences.append(id)
    occurrences.sort()

    return occurrences


def load_and_process_trans(file_path, tokens_to_exclude, lower_case=True):
    """
    Function to load and process the texts (transcriptions annotated for the SZTE-Dementia dataset)
    Args:

    """
    df = pd.read_csv(file_path, sep='\n', header=None, encoding='unicode_escape')
    df.columns = ['token']

    if tokens_to_exclude:
        indices_to_exclude = []
        for i in tokens_to_exclude:
            indices = df.index[df['token'] == i].tolist()
            indices_to_exclude.append(indices)

        # squeezing the list of lists
        indices_to_exclude = list(np.concatenate(indices_to_exclude))
        indices_to_exclude.sort()
        indices_to_exclude = np.asarray(indices_to_exclude, dtype=int)
        # dropping tokens from the original text
        no_marks_df = df.drop(df.index[indices_to_exclude])
        # serializing dataframe for lower-casing
        serialized_df = no_marks_df.squeeze()
        if lower_case:
            serialized_df = serialized_df.str.lower()
            final_transcription = serialized_df.str.cat(sep=" ")
        else:
            final_transcription = serialized_df.str.cat(sep=" ")

        return final_transcription
    else:
        # serializing dataframe for lower-casing
        serialized_df = df.squeeze()
        if lower_case:
            serialized_df = serialized_df.str.lower()
            final_transcription = serialized_df.str.cat(sep=" ")
        else:
            final_transcription = serialized_df.str.cat(sep=" ")

        return final_transcription


def replace_tokens(df, original_tokens, new_tokens):
    df = df.replace(to_replace=original_tokens, value=new_tokens)

    return df


def my_collate(batch):
    """
    Function used to pad the sequences (text samples) so they have the same size.
    Args:
        batch: The batch containing the samples.

    Returns:
        paded data, labels
    """
    # batch contains a list of tuples of structure (sequence, target)
    data = [item['embeddings'] for item in batch]
    data = pack_sequence(data, enforce_sorted=False)
    targets = [item['label'] for item in batch]
    return [data, targets]

# utils for DEMENTIA SZTE DATASET ###
