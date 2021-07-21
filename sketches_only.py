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
import numpy as np

from feature_extraction import get_feats
import train_utils
from CustomDataset import CustomAudioDataset

# task (name of the dataset)
task = 'sleepiness'
# in and out dirs
corpora_dir = '/media/jose/hk-data/PycharmProjects/the_speech/audio/'
out_dir = 'data/' + task
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

# Get model params
parser = train_utils.getParams()
args = parser.parse_args()

# Loading the data
# train_set = CustomDataset(file_labels=labels, audio_dir=task_audio_dir, name_set='train', online=True)
train_set = CustomAudioDataset(file_labels='data/sleepiness/labels/labels.csv', audio_dir=task_audio_dir,
                               name_set='train', online=True,
                               # feats_info=['/media/jose/hk-data/PycharmProjects/the_speech/data/sleepiness/', 'mfcc'],
                               calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir, feat_type='mfcc',
                                                                 deltas=1, **params)
                               )

# dev_set = CustomDataset(file_labels='data/sleepiness/labels/labels.csv', audio_dir=task_audio_dir, name_set='dev',
#                               online=False,
#                               feats_info=['/media/jose/hk-data/PycharmProjects/the_speech/data/sleepiness/', 'mfcc'],
#                               #         calc_flevel=get_feats.FbanksKaltorch(**params)
#                               )
# dev_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=False, num_workers=0)
#
# test_set = CustomDataset(file_labels='data/sleepiness/labels/labels.csv', audio_dir=task_audio_dir, name_set='test',
#                               online=False,
#                               feats_info=['/media/jose/hk-data/PycharmProjects/the_speech/data/sleepiness/', 'mfcc'],
#                               #         calc_flevel=get_feats.FbanksKaltorch(**params)
#                               )
# test_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=False, num_workers=0)

# Set the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set Params
total_steps = args.numEpochs #* args.numArchives
numBatchesPerArk = int(args.numEgsPerFile/args.batchSize)
eps = args.noiseEps
# prepare the model
net, optimizer, step, saveDir = train_utils.prepare_model(args)
criterion = nn.CrossEntropyLoss()
# LR scheduler
cyclic_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                          max_lr=args.maxLR,
                                                          cycle_momentum=False,
                                                          div_factor=5,
                                                          final_div_factor=1e+3,
                                                          total_steps=total_steps, # * numBatchesPerArk,
                                                          pct_start=0.15)

# Train model
while step < total_steps:
    preFetchRatio = args.preFetchRatio

    batchI, loggedBatch = 0, 0
    loggingLoss = 0.0
    start_time = time.time()
    for _, sample_batched in enumerate(train_loader):
        x_train = sample_batched['feature'].to(device)
        x_train = torch.transpose(x_train, 1, -1)
        y_train = sample_batched['label'].to(device)
    accumulateStepSize = 4
    preFetchBatchI = 0  # this counter within the prefetched batches only
    while preFetchBatchI < int(len(y_train) / args.batchSize) - accumulateStepSize:
        # zeroing the gradients
        optimizer.zero_grad()
        for _ in range(accumulateStepSize):
            batchI += 1
            preFetchBatchI += 1
            # forward prop + backward prop + optimization
            output = net(x_train, args.numEpochs)
            loss = criterion(output, y_train)
            if np.isnan(loss.item()):  # checking for exploding gradient
                print('Nan encountered at iter %d. Exiting...' % iter)
                sys.exit(1)
            loss.backward()
            # stats
            loggingLoss += loss.item()
        optimizer.step()  # updating weights
        cyclic_lr_scheduler.step()

        print(batchI, loggedBatch, args.logStepSize)

        # Log
        if batchI - loggedBatch >= args.logStepSize:
            logStepTime = time.time() - start_time
            print('Batch: (%d/%d)     Avg Time/batch: %1.3f      Avg Loss/batch: %1.3f' % (
                batchI,
                numBatchesPerArk,
                logStepTime / (batchI - loggedBatch),
                loggingLoss / (batchI - loggedBatch)))
            loggingLoss = 0.0
            start_time = time.time()
            loggedBatch = batchI

# for args.numEpochs in range(args.numEpochs):
#     loggingLoss = 0.0
#     start_time = time.time()
#     # net.train()
#     for batch_idx, sample_batched in enumerate(train_loader):
#         x_train = sample_batched['feature'].to(device)
#         x_train = torch.transpose(x_train, 1, -1)
#         y_train = sample_batched['label'].to(device)
#         # zeroing the gradients
#         optimizer.zero_grad()
#         # forward prop + backward prop + optimization
#         output = net(x_train, args.numEpochs)
#         loss = criterion(output, y_train)
#         if np.isnan(loss.item()):  # checking for exploding gradient
#             print('Nan encountered at iter %d. Exiting...' % iter)
#             sys.exit(1)
#         loss.backward()
#         # stats
#         loggingLoss += loss.item()
#         optimizer.step()  # updating weights
#         cyclic_lr_scheduler.step()
#
#         # if batch_idx % 20 == 0:
#         logTime = time.time() - start_time
#         print(loggingLoss)
#         # print('Batch: %d    Avg time/batch: %1.3f   Avg loss/batch: %1.3f' %(
#         #         batch_idx, logTime/batch_idx, loggingLoss/300))


