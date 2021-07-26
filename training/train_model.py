"""
Created by José Vicente Egas López
on 2021. 02. 04. 13 10

"""
import sys

from sklearn.metrics import recall_score

sys.path.extend(['/home/jose/PycharmProjects/dnn_embeddings_pytorch'])

import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from feature_extraction import get_feats
import train_utils
from CustomDataset import CustomAudioDataset

# task (name of the dataset)
task = 'sleepiness'
# in and out dirs
corpora_dir = '/media/jvel/data/audio/'
out_dir = 'data/' + task
task_audio_dir = corpora_dir + task + '/'
# labels
labels = 'data/{}/labels/labels.csv'.format(task)

# Get model params
parser = train_utils.get_train_params(task=task, flevel='mfcc')
args = parser.parse_args()
if not args.online and (args.feat_dir_train is None or args.feat_dir_dev is None):
    parser.error("When -online=False, please specify -feat_dir_train and -feat_dir_dev.")

# Loading the data
train_set = CustomAudioDataset(file_labels=args.labels, audio_dir=task_audio_dir, online=args.online,
                               feats_fir=args.feats_dir_train, max_length_sec=1,
                               calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir,
                                                                 feat_type=args.feat_type,
                                                                 deltas=args.deltas, config_file=args.config_file)
                               )
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, drop_last=False, pin_memory=True)

dev_set = CustomAudioDataset(file_labels=args.labels, audio_dir=task_audio_dir, online=args.online,
                             feats_fir=args.feats_dir_dev, max_length_sec=1,
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

test_set = CustomAudioDataset(file_labels=args.labels, audio_dir=task_audio_dir, online=args.online,
                              feats_fir=args.feats_dir_test, max_length_sec=1,
                              calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir,
                                                                feat_type=args.feat_type,
                                                                deltas=args.deltas, config_file=args.config_file)
                              )
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False,
                         num_workers=0, drop_last=False, pin_memory=True)


# Decay LR by a factor of 0.1 every 7 epochs
# cyclic_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train model
def train_model(data_loader_train, data_loader_eval, num_epochs):
    since = time.time()
    # Set the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # prepare the model
    net, optimizer, epoch_n, save_dir, loss = train_utils.prepare_model(args)
    criterion = nn.BCEWithLogitsLoss()
    # LR scheduler
    cyclic_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                              max_lr=args.max_LR,
                                                              cycle_momentum=False,
                                                              div_factor=5,
                                                              final_div_factor=1e+3,
                                                              total_steps=args.num_epochs * len(data_loader_train),
                                                              # * numBatchesPerArk,
                                                              pct_start=0.15)
    best_loss = loss
    num_epochs = num_epochs - epoch_n  # validating number of epochs when resume training

    for epoch in range(num_epochs):
        # set model to train phase
        net.train()
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        train_logging_loss = 0.0
        val_logging_loss = 0.0

        for batch_idx, sample_batched in enumerate(data_loader_train):
            x_train = sample_batched['feature'].to(device)
            x_train = torch.transpose(x_train, 1, -1).unsqueeze(1)
            # print(x_train.shape)
            y_train = sample_batched['label'].to(device)
            y_train = y_train.to(dtype=torch.long)

            optimizer.zero_grad()  # zeroing the gradients
            output_logits, output = net(x_train)  # forward prop + backward prop + optimization
            loss = criterion(output_logits, y_train.unsqueeze(1).float())
            loss.backward()

            # stats
            train_logging_loss += loss.item() * y_train.shape[0]
            optimizer.step()  # updating weights

            cyclic_lr_scheduler.step()
        # uar = recall_score(np.hstack(truths_list), np.hstack(preds_list), average='macro')
        train_epoch_loss = train_logging_loss / len(data_loader_train.dataset)
        print('Loss: {:.4f}'.format(train_epoch_loss))

        # EVALUATION: set model to dev phase
        net.eval()
        preds_list = []
        truths_list = []

        for batch_idx, sample_batched in enumerate(data_loader_eval):
            x_dev = sample_batched['feature'].to(device)
            x_dev = torch.transpose(x_dev, 1, -1).unsqueeze(1)
            y_dev = sample_batched['label'].to(device)
            y_dev = y_dev.to(dtype=torch.long)

            output_logits, output = net(x_dev)
            loss = criterion(output_logits, y_dev.unsqueeze(1).float())
            val_logging_loss += loss.item() * y_dev.shape[0]
            preds = output > 0.5

            preds_list.append(preds.squeeze().cpu().detach().numpy())
            truths_list.append(y_dev.cpu().detach().numpy())
        uar = recall_score(np.hstack(truths_list), np.hstack(preds_list), average='macro')
        val_loss = val_logging_loss / len(data_loader_eval.dataset)
        print("Validation:")
        print('Loss: {:.4f} - UAR: {}'.format(val_loss, uar))

        if val_loss < best_loss:
            best_loss = val_loss
            # best_model_wts = copy.deepcopy(net.state_dict())
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': best_loss,
            }, '{}/checkpoint_{}'.format(save_dir, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))


if __name__ == '__main__':
    train_model(data_loader_train=train_dev_loader, data_loader_eval=test_loader, num_epochs=args.num_epochs)
