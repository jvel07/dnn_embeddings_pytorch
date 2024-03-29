"""
Created by José Vicente Egas López
on 2021. 03. 03. 16 54

"""

from sklearn.metrics import recall_score
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch
import torch.nn as nn

import numpy as np

import train_utils
from CustomDataset import CustomAudioDataset
from feature_extraction import get_feats

# task (name of the dataset)
task = 'mask'
# in and out dirs
corpora_dir = '/media/jose/hk-data/PycharmProjects/the_speech/audio/'
out_dir = 'data/' + task
task_audio_dir = corpora_dir + task + '/'

# Get params
parser = train_utils.get_train_params(task, 'mfcc')
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


# Defining the pretrained model
def use_resnet(net_output, binary_class=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet_model = resnet18(pretrained=True)
    # num_ftrs = resnet_model.fc.in_features

    # freeze pretrained layers
    for param in resnet_model.parameters():
        param.requires_grad = False
    # adapting number of channels from 3 (originally) to 1
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # redefine/adapt final fc layer
    if binary_class:
        net_output = 1
        resnet_model.fc = nn.Linear(512, net_output)
        # resnet_model.out = nn.Sequential(nn.Linear(2048, 512),
        #                                  nn.ReLU(),
        #                                  nn.Dropout(0.3),
        #                                  nn.Linear(512, net_output),
        #                                  nn.Sigmoid()
                                         # )
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(resnet_model.parameters(), lr=args.base_LR)
        criterion = nn.BCEWithLogitsLoss()
    else:
        resnet_model.out = nn.Sequential(nn.Linear(2048, 512),
                                         nn.ReLU(),
                                         nn.Dropout(0.3),
                                         nn.Linear(512, net_output),
                                         nn.Softmax(dim=1)
                                         )
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(resnet_model.parameters(), lr=args.base_LR)
        criterion = nn.CrossEntropyLoss()

    resnet_model = resnet_model.to(device)

    return resnet_model, device, optimizer, criterion


# Train model
def train_model(train_data_loader, eval_data_loader, num_epochs):
    # Instantiating resnet
    net, device, optimizer, criterion = use_resnet(args.net_output, True)
    # or load the resnet re-trained on customized data
    # code for loading pending...

    # LR scheduler
    cyclic_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                              max_lr=args.max_LR,
                                                              cycle_momentum=False,
                                                              div_factor=5,
                                                              final_div_factor=1e+3,
                                                              total_steps=args.num_epochs * len(train_data_loader),
                                                              pct_start=0.15)

    best_loss = 10

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        net.train()  # set model to train phase
        train_logging_loss = 0.0
        resnet_train_losses = []
        preds_list = []
        truths_list = []
        batch_losses = []

        for batch_idx, sample_batched in enumerate(train_data_loader):
            batch_losses = []
            x_train = sample_batched['feature'].to(device).unsqueeze(1)  # .squeeze()
            # x_train = torch.transpose(x_train, 1, -1).unsqueeze(1)
            y_train = sample_batched['label'].to(device)
            # print(x_train.shape)

            # forward prop + backward prop + optimization
            optimizer.zero_grad()
            output = net(x_train)

            loss = criterion(output, y_train.unsqueeze(1).float())
            loss.backward()
            optimizer.step()  # updating weights

            # stats
            train_logging_loss += loss.item() * y_train.shape[0]

        cyclic_lr_scheduler.step()
        train_epoch_loss = train_logging_loss / len(train_data_loader.dataset)
        print('Loss: {:.4f}'.format(train_epoch_loss))

        # print(f'Epoch - {epoch} / {num_epochs-1} Train-Loss : {np.mean(resnet_train_losses[-1])} '
        #       f'UAR: {uar} - epoch-loss: {epoch_loss}')

        # EVALUATION: set model to dev phase
        net.eval()
        preds_list = []
        truths_list = []
        val_logging_loss = 0.0

        for batch_idx, sample_batched in enumerate(eval_data_loader):
            x_dev = sample_batched['feature'].to(device).unsqueeze(1)
            # x_dev = torch.transpose(x_dev, 1, -1).unsqueeze(1)
            y_dev = sample_batched['label'].to(device)
            y_dev = y_dev.to(dtype=torch.long)

            output = net(x_dev)
            loss = criterion(output, y_dev.unsqueeze(1).float())
            val_logging_loss += loss.item() * y_dev.shape[0]

            # getting probs
            # m = nn.Sigmoid()
            # probs = m(output)
            preds = output > 0.5
            # print(y_dev)
            # print(preds)

            preds_list.append(preds.squeeze().cpu().detach().numpy())
            truths_list.append(y_dev.cpu().detach().numpy())
        uar = recall_score(np.hstack(truths_list), np.hstack(preds_list), average='macro')
        val_loss = val_logging_loss / len(eval_data_loader.dataset)
        print("Validation:")
        print('Loss: {:.4f} - UAR: {}'.format(val_loss, uar))


if __name__ == '__main__':
    train_model(train_dev_loader, test_loader, 32)
