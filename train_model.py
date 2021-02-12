"""
Created by José Vicente Egas López
on 2021. 02. 04. 13 10

"""
import sys

sys.path.extend(['/home/jose/PycharmProjects/dnn_embeddings_pytorch'])

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

# Get model params
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-feat_type', type=str, default='spectrogram', help="Type of the frame-level features to load or"
                                                                        "extract. Available types: mfcc, fbanks, spec,"
                                                                        "melspecT")
parser.add_argument('-config_file', default='conf/spectrogram.ini', help='Path to the config (ini) file.')
parser.add_argument('-deltas', default=0, type=int, help="Compute delta coefficients of a tensor. '1' for "
                                                         "first order derivative, '2' for second order. "
                                                         "None for not using deltas. Default: None.")

parser.add_argument('-online', default=True, required=False, help='If True, features are computed on the fly. '
                                                                  'If False, features are loaded from disk from '
                                                                  'specified input of -. Default: True')
parser.add_argument('-labels', default='data/sleepiness/labels/labels.csv', required=False, help='Path to the file '
                                                                                                 'containing the labels.')
parser.add_argument('-feats_dir_train', default='data/sleepiness/spectrogram/train',
                    help='Path to the folder containing the features.')
parser.add_argument('-feats_dir_dev', default='data/sleepiness/spectrogram/dev',
                    help='Path to the folder containing the features.')

parser.add_argument('-model_out_dir', default='data/sleepiness/spectrogram',
                    help='Path to the folder containing the features.')

parser.add_argument('-input_dim', action="store_true", default=257)
parser.add_argument('-num_classes', action="store_true", default=10)
parser.add_argument('-batch_size', action="store_true", default=128)
parser.add_argument('-num_epochs', action="store_true", default=100)

parser.add_argument('-training_mode', default='init',
                    help='(init) Train from scratch, (resume) Resume training, (finetune) Finetune a pretrained model')
parser.add_argument('-model_type', default='xvecTDNN', help='Model class. Check models.py')
parser.add_argument('-base_LR', default=1e-3, type=float, help='Initial LR')
parser.add_argument('-max_LR', default=2e-3, type=float, help='Maximum LR')

args = parser.parse_args()

if not args.online and (args.feat_dir_train is None or args.feat_dir_dev is None):
    parser.error("When -online=False, please specify -feats_dir.")

# Loading the data
train_set = CustomDataset(file_labels=args.labels, audio_dir=task_audio_dir, online=args.online,
                          feats_fir=args.feats_dir_train,
                          calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir,
                                                                 feat_type=args.feat_type,
                                                                 deltas=args.deltas, config_file=args.config_file)
                          )
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, drop_last=False, pin_memory=True)

dev_set = CustomDataset(file_labels=args.labels, audio_dir=task_audio_dir, online=args.online,
                        feats_fir=args.feats_dir_dev,
                        calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir,
                                                               feat_type=args.feat_type,
                                                               deltas=args.deltas, config_file=args.config_file)
                        )
dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False, pin_memory=True)

# Concatenating Datasets for training with Train and Dev
train_dev_sets = torch.utils.data.ConcatDataset([train_set, dev_set])
train_dev_loader = DataLoader(dataset=train_dev_sets, batch_size=args.batch_size, shuffle=False, num_workers=0,
                              drop_last=False, pin_memory=True)

# Set the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# prepare the model
net, optimizer, step, save_dir = train_utils.prepare_model(args)
criterion = nn.CrossEntropyLoss()
# LR scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=args.max_LR,
                                                       cycle_momentum=False,
                                                       div_factor=5,
                                                       final_div_factor=1e+3,
                                                       total_steps=args.num_epochs * len(train_dev_loader),
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
            # best_model_wts = copy.deepcopy(net.state_dict())
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, '{}/checkpoint_{}'.format(save_dir, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net


if __name__ == '__main__':
    train_model(data_loader=train_dev_loader, num_epochs=args.num_epochs)
