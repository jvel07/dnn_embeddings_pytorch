# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torchaudio
from torch.utils.data import DataLoader

from SleepinessDataset import SleepinessDataset

"""
Created by José Vicente Egas López
on 2021. 01. 27. 11 57

"""

# task (name of the dataset)
task = 'sleepiness'
# in and out dirs
corpora_dir = '/media/jose/hk-data/PycharmProjects/the_speech/audio/'
out_dir = 'data/'
task_audio_dir = corpora_dir + task + '/'
# labels
labels = 'data/sleepiness/labels/labels.csv'

# List of audio-sets inside 'task_audio_dir' (folder(s) containing audio samples) i.e. train, dev, test...
list_sets = ['train', 'dev', 'test']

# frame-level feats params
params = {
    "num_mel_bins": 40
}

# Loading the data
# train_set = SleepinessDataset(file_labels=labels, audio_dir=task_audio_dir, name_set='train', online=True)
train_set = SleepinessDataset('data/sleepiness/labels/labels.csv', task_audio_dir, 'train', online=False,
                      feats_info=['/media/jose/hk-data/PycharmProjects/the_speech/data/sleepiness/', 'mfcc'])
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=False)
