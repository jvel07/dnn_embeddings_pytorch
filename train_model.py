"""
Created by José Vicente Egas López
on 2021. 02. 04. 13 10

"""
import sys

from sklearn.metrics import recall_score

sys.path.extend(['/home/jose/PycharmProjects/dnn_embeddings_pytorch'])

import argparse
import copy
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from feature_extraction import get_feats
import train_utils
from CustomDataset import CustomDataset

# task (name of the dataset)
task = 'sleepiness'
# in and out dirs
corpora_dir = '/media/jose/hk-data/PycharmProjects/the_speech/audio/'
out_dir = 'data/' + task
task_audio_dir = corpora_dir + task + '/'
# labels
labels = 'data/sleepiness/labels/labels_orig.csv'

# Get model params
parser = train_utils.get_train_params()
args = parser.parse_args()
if not args.online and (args.feat_dir_train is None or args.feat_dir_dev is None):
    parser.error("When -online=False, please specify -feat_dir_train adn -feat_dir_dev.")

# Loading the data
train_set = CustomDataset(file_labels=args.labels, audio_dir=task_audio_dir, online=args.online,
                          feats_fir=args.feats_dir_train, max_length_sec=25,
                          calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir,
                                                                 feat_type=args.feat_type,
                                                                 deltas=args.deltas, config_file=args.config_file)
                          )
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, drop_last=False, pin_memory=True)

dev_set = CustomDataset(file_labels=args.labels, audio_dir=task_audio_dir, online=args.online,
                        feats_fir=args.feats_dir_dev, max_length_sec=25,
                        calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir,
                                                               feat_type=args.feat_type,
                                                               deltas=args.deltas, config_file=args.config_file)
                        )
dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False, pin_memory=True)

# Concatenating Datasets for training with Train and Dev
# train_dev_sets = torch.utils.data.ConcatDataset([train_set, dev_set])
# train_dev_loader = DataLoader(dataset=train_dev_sets, batch_size=args.batch_size, shuffle=False, num_workers=0,
#                               drop_last=False, pin_memory=True)
# n_iters = 3000
# num_epochs = n_iters / (len(train_dev_sets) / args.batch_size)


# Set the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# prepare the model
net, optimizer, step, save_dir = train_utils.prepare_model(args)
criterion = nn.CrossEntropyLoss()
# LR scheduler
cyclic_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                          max_lr=args.max_LR,
                                                          cycle_momentum=False,
                                                          div_factor=5,
                                                          final_div_factor=1e+3,
                                                          total_steps=args.num_epochs * len(train_loader),
                                                          # * numBatchesPerArk,
                                                          pct_start=0.15)


# Decay LR by a factor of 0.1 every 7 epochs
# cyclic_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# Train model
def train_model(data_loader, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = 10.0
    iter = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # set model to train phase
        net.train()
        logging_loss = 0.0
        preds_list = []
        truths_list = []
        loss_list = []

        for batch_idx, sample_batched in enumerate(data_loader):
            x_train = sample_batched['feature'].to(device)
            x_train = torch.transpose(x_train, 1, -1).unsqueeze(1)
            print(x_train.shape)
            y_train = sample_batched['label'].to(device)
            # y_train = y_train.to(dtype=torch.long)
            optimizer.zero_grad()  # zeroing the gradients
            output = net(x_train)  # forward prop + backward prop + optimization
            preds = torch.argmax(output, dim=1)
            # adding preds and ground truths to the lists
            preds_list.append(preds.cpu().detach().numpy())
            truths_list.append(y_train.cpu().detach().numpy())
            loss = criterion(output, y_train)
            loss.backward()
            # stats
            logging_loss += loss.item() #* y_train.shape[0]
            optimizer.step()  # updating weights
        # iter += 1
        # if iter % 500 == 0:
        #     print('Loss: {:.4f}'.format(epoch_loss))
        # loss_list.append(logging_loss / len(data_loader.dataset))
        cyclic_lr_scheduler.step()
        uar = recall_score(truths_list, preds_list, average='macro')
        epoch_loss = logging_loss / len(data_loader.dataset)
        print('Loss: {:.4f} - UAR: {}'.format(epoch_loss, uar / len(data_loader.dataset)))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # best_model_wts = copy.deepcopy(net.state_dict())
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, '{}/checkpoint_{}'.format(save_dir, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    # net.load_state_dict(best_model_wts)
    # return net


def eval_model(data_loader, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = 10.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # set model to dev phase
        net.eval()
        logging_loss = 0.0
        preds_list = []
        truths_list = []
        loss_list = []

        for batch_idx, sample_batched in enumerate(data_loader):
            x_dev = sample_batched['feature'].to(device)
            x_dev = torch.transpose(x_dev, 1, -1)
            y_dev = sample_batched['label']
            # y_train = y_train.to(dtype=torch.long)
            y_dev = y_dev.to(device)
            output = net(x_dev)
            loss = criterion(output, y_dev)
            logging_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            preds_list.append(preds.cpu().detach().numpy())
            truths_list.append(y_dev.cpu().detach().numpy())
        uar = recall_score(truths_list, preds_list, average='macro')
        epoch_loss = logging_loss / len(data_loader.dataset)
        print('Loss: {:.4f} - UAR: {}'.format(epoch_loss, uar / len(data_loader.dataset)))

        return preds_list, truths_list


if __name__ == '__main__':
    train_model(data_loader=train_loader, num_epochs=args.num_epochs)
    preds_list, truths_list = eval_model(data_loader=dev_loader, num_epochs=args.num_epochs)
