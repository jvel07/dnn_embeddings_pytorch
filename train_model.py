"""
Created by José Vicente Egas López
on 2021. 02. 04. 13 10

"""
import argparse
import copy
import sys
import time

import torch
import torch.nn as nn
import torchaudio
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchaudio import transforms
import numpy as np

import dnn_models
import get_feats
import train_utils
import utils
from CustomDataset import CustomDataset

# task (name of the dataset)
task = 'sleepiness'
# in and out dirs
corpora_dir = '/media/jose/hk-data/PycharmProjects/the_speech/audio/'
out_dir = 'data/' + task
task_audio_dir = corpora_dir + task + '/'
# labels
labels = 'data/sleepiness/labels/labels.csv'

# frame-level feats params/config
params = utils.read_conf_file(file_name='spectrogram.ini', conf_section='DEFAULTS')

# Get model params
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-feat_type', type=str, default='mfcc', help="Type of the frame-level features to load or "
                                                                 "extract. Available types: mfcc, fbanks, spec, "
                                                                 "melspecT")

parser.add_argument('-training_filepath', type=str, default='meta/training_feat.txt')
parser.add_argument('-testing_filepath', type=str, default='meta/testing_feat.txt')
parser.add_argument('-validation_filepath', type=str, default='meta/validation_feat.txt')

parser.add_argument('-input_dim', action="store_true", default=257)
parser.add_argument('-num_classes', action="store_true", default=10)
parser.add_argument('-lamda_val', action="store_true", default=0.1)
parser.add_argument('-batch_size', action="store_true", default=128)
parser.add_argument('-use_gpu', action="store_true", default=True)
parser.add_argument('-num_epochs', action="store_true", default=100)

parser.add_argument('-trainingMode', default='init',
                    help='(init) Train from scratch, (resume) Resume training, (finetune) Finetune a pretrained model')
parser.add_argument('-modelType', default='xvecTDNN', help='Model class. Check models.py')
parser.add_argument('-baseLR', default=1e-3, type=float, help='Initial LR')
parser.add_argument('-maxLR', default=2e-3, type=float, help='Maximum LR')

args = parser.parse_args()

# Loading the data
# train_set = CustomDataset(file_labels=labels, audio_dir=task_audio_dir, name_set='train', online=True)
train_set = CustomDataset(file_labels='data/sleepiness/labels/labels.csv', audio_dir=task_audio_dir,
                          name_set='train', online=True,
                          # feats_info=['data/sleepiness/{}/'.format(args.feat_type), '.npy'],
                          calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir,
                                                                 feat_type='spectrogram',
                                                                 deltas=0, **params)
                          )
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, drop_last=False)

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
# prepare the model
net, optimizer, step, saveDir = train_utils.prepare_model(args)
criterion = nn.CrossEntropyLoss()
# LR scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=args.maxLR,
                                                       cycle_momentum=False,
                                                       div_factor=5,
                                                       final_div_factor=1e+3,
                                                       total_steps=args.num_epochs * len(train_loader),
                                                       # * numBatchesPerArk,
                                                       pct_start=0.15)


# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# Train model
def train_model(data_loader, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = 10.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        net.train()
        logging_loss = 0

        for batch_idx, sample_batched in enumerate(data_loader):
            x_train = sample_batched['feature'].to(device)
            x_train = torch.transpose(x_train, 1, -1)
            y_train = sample_batched['label'].to(device)
            # zeroing the gradients
            optimizer.zero_grad()
            # forward prop + backward prop + optimization
            output = net(x_train, args.num_epochs)
            loss = criterion(output, y_train)
            loss.backward()
            # stats
            logging_loss += loss.item() * y_train.shape[0]
            optimizer.step()  # updating weights
        exp_lr_scheduler.step()
        epoch_loss = logging_loss / len(train_set)

        print('Loss: {:.4f}'.format(epoch_loss))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(net.state_dict())
            # Save checkpoint
            torch.save({
                'step': step,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
            }, '{}/checkpoint_step{}.tar'.format(saveDir, step))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net


if __name__ == '__main__':
    train_model(data_loader=train_loader, num_epochs=50)
