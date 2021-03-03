"""
Created by José Vicente Egas López
on 2021. 03. 03. 16 54

"""
import time

from sklearn.metrics import recall_score
from torch.utils.data import DataLoader
from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import train_utils
from CustomDataset import CustomDataset
from feature_extraction import get_feats

# task (name of the dataset)
task = 'sleepiness'
# in and out dirs
corpora_dir = '/media/jose/hk-data/PycharmProjects/the_speech/audio/'
out_dir = 'data/' + task
task_audio_dir = corpora_dir + task + '/'

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


# Defining the pretrained model
def use_resnet34(num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet_model = resnet34(pretrained=True)
    # adapting number of classes
    resnet_model.fc = nn.Linear(512, num_classes)
    # adapting number of channels to 1
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet_model = resnet_model.to(device)

    return resnet_model, device


# Instantiating resnet
net, device = use_resnet34(args.num_classes)

# Optimizer and scheduler
optimizer = torch.optim.Adam(net.parameters(), lr=args.base_LR)
criterion = nn.CrossEntropyLoss()
# LR scheduler
cyclic_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                          max_lr=args.max_LR,
                                                          cycle_momentum=False,
                                                          div_factor=5,
                                                          final_div_factor=1e+3,
                                                          total_steps=args.num_epochs * len(train_loader),
                                                          pct_start=0.15)


# Train model
def train_model(_train_loader, _dev_loader, num_epochs):

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # set model to train phase
        net.train()
        logging_loss = 0.0
        resnet_train_losses = []
        resnet_valid_losses = []
        batch_losses = []

        for batch_idx, sample_batched in enumerate(_train_loader):
            batch_losses = []
            x_train = sample_batched['feature'].to(device)#.squeeze()
            # x_train = torch.transpose(x_train, 1, -1)#.unsqueeze(1)
            y_train = sample_batched['label'].to(device)
            optimizer.zero_grad()
            output = net(x_train)  # forward prop + backward prop + optimization
            loss = criterion(output, y_train)
            loss.backward()
            batch_losses.append(loss.item())  # stats
            optimizer.step()  # updating weights
        resnet_train_losses.append(batch_losses)
        print(f'Epoch - {epoch} Train-Loss : {np.mean(resnet_train_losses[-1])}')

        # set model to eval phase
        net.eval()
        batch_losses = []
        preds_list = []
        truths_list = []
        for batch_idx, sample_batched in enumerate(_dev_loader):
            x_dev = sample_batched['feature'].to(device)
            # x_dev = torch.transpose(x_dev, 1, -1)#.unsqueeze(1)
            y_dev = sample_batched['label'].to(device)
            optimizer.zero_grad()
            output = net(x_dev)  # forward prop + backward prop + optimization
            preds = torch.argmax(output, dim=1)
            preds_list.append(preds.cpu().detach().numpy())
            truths_list.append(y_dev.cpu().detach().numpy())
            loss = criterion(output, y_dev)
            batch_losses.append(loss.item())  # stats
        resnet_valid_losses.append(batch_losses)
        preds_list = np.concatenate(preds_list)
        truths_list = np.concatenate(truths_list)
        uar = recall_score(truths_list, preds_list, average='macro')
        print(f'Epoch - {epoch} Valid-Loss : {np.mean(resnet_valid_losses[-1])} Valid-Accuracy : {uar}')


if __name__ == '__main__':
    train_model(train_loader, dev_loader, 50)


