"""
Created by José Vicente Egas López
on 2021. 01. 27. 11 57

"""

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import time

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from torchaudio import transforms
import numpy as np

import dnn_models
import get_feats
import train_utils
from SleepinessDataset import SleepinessDataset

# task (name of the dataset)
task = 'sleepiness'
# in and out dirs
corpora_dir = '/media/jose/hk-data/PycharmProjects/the_speech/audio/'
out_dir = 'data/'
task_audio_dir = corpora_dir + task + '/'
# labels
labels = 'data/sleepiness/labels/labels.csv'

# frame-level feats params
# params = {
#     sample-frequency : 16000,
#     frame-length: 25, # the default is 25
#     low-freq: 20, # the default.
#     high-freq: 0, # the default is zero meaning use the Nyquist (4k in this case).
#     num-ceps: 20, # higher than the default which is 12.
#     snip-edges: "false"
# }


params = {
    "num_mel_bins": 40
}

# params
parser = train_utils.getParams()
args = parser.parse_args()
totalSteps = args.numEpochs

# Loading the data
# train_set = SleepinessDataset(file_labels=labels, audio_dir=task_audio_dir, name_set='train', online=True)
train_set = SleepinessDataset(file_labels='data/sleepiness/labels/labels.csv', audio_dir=task_audio_dir,
                              name_set='train', online=True,
                              # feats_info=['/media/jose/hk-data/PycharmProjects/the_speech/data/sleepiness/', 'mfcc'],
                              transform=get_feats.FbanksKaltorch(**params)
                              )
train_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=False, num_workers=0)

# dev_set = SleepinessDataset(file_labels='data/sleepiness/labels/labels.csv', audio_dir=task_audio_dir, name_set='dev',
#                               online=False,
#                               feats_info=['/media/jose/hk-data/PycharmProjects/the_speech/data/sleepiness/', 'mfcc'],
#                               #         transform=get_feats.FbanksKaltorch(**params)
#                               )
# dev_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=False, num_workers=0)
#
# test_set = SleepinessDataset(file_labels='data/sleepiness/labels/labels.csv', audio_dir=task_audio_dir, name_set='test',
#                               online=False,
#                               feats_info=['/media/jose/hk-data/PycharmProjects/the_speech/data/sleepiness/', 'mfcc'],
#                               #         transform=get_feats.FbanksKaltorch(**params)
#                               )
# test_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=False, num_workers=0)

# Prepare the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net, optimizer, step, saveDir = train_utils.prepare_model(args)
criterion = nn.CrossEntropyLoss()

# LR SCHEDULERS
cyclic_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                          max_lr=args.maxLR,
                                                          cycle_momentum=False,
                                                          div_factor=5,
                                                          final_div_factor=1e+3,
                                                          total_steps=totalSteps,
                                                          pct_start=0.15)

# train model
# net.train()
for batch_idx, (X, Y) in enumerate(train_loader):
    x_train = X.to(device)
    y_train = Y.to(device)
# optimizer.zero_grad()
