"""
Created by José Vicente Egas López
on 2021. 02. 01. 13 00

"""
import argparse
import glob
import os
from datetime import datetime
# from prettytable import PrettyTable

import kaldi_python_io
import numpy as np
from torch import nn
from torchvision import models

import dnn_models

import torch


def prepare_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.training_mode == 'init':
        print('Initializing Model...')
        epoch = 0
        loss = 10.0
        net = eval('dnn_models.{0}({1}, {2}, p_dropout=0.4)'.format(args.model_type, args.input_dim, args.net_output))
        print(net)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.base_LR)
        event_ID = datetime.now().strftime('%Y%m-%d%H-%M%S')
        save_dir = '{}/models/modelType_{}_event_{}'.format(args.model_out_dir, args.model_type, event_ID)
        os.makedirs(save_dir)

        #return net, optimizer, step, save_dir

    elif args.training_mode == 'resume':
        print('Loading trained model...')
        # select the latest model from modelDir
        model_file = max(glob.glob(args.model_out_dir+'/*'), key=os.path.getctime)
        net = eval('dnn_models.{0}({1}, {2}, p_dropout=0.4)'.format(args.model_type, args.input_dim, args.net_output))
        optimizer = torch.optim.Adam(net.parameters(), lr=args.base_LR)
        # Load model params
        checkpoint = torch.load(model_file, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        save_dir = args.mmodel_out_dir

    return net.to(device), optimizer, epoch, save_dir, loss


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_train_params(task, flevel):
    parser = argparse.ArgumentParser()
    parser.add_argument('-audio_dir', default='/media/jose/hk-data/PycharmProjects/the_speech/audio/{}/'.format(task),
                        help='Path to the folder containing the wavs.')

    parser.add_argument('-feat_type', type=str, default=flevel, help="Type of the frame-level features to load or"
                                                                     "extract. Available types: mfcc, fbanks, spec,"
                                                                     "melspecT")
    parser.add_argument('-config_file', default='conf/{}.ini'.format(flevel), help='Path to the config (ini) file.')
    parser.add_argument('-deltas', default=0, type=int, help="Compute delta coefficients of a tensor. '1' for "
                                                             "first order derivative, '2' for second order. "
                                                             "None for not using deltas. Default: None.")

    parser.add_argument('-online', default=True, required=False, help='If True, features are computed on the fly. '
                                                                      'If False, features are loaded from disk from '
                                                                      'specified input of -feats_dir_train/ ..dev .. '
                                                                      '/test. Default: True')
    parser.add_argument('-labels', default='data/{}/labels'.format(task), required=False,
                        help='Path to the folder containing the files with the labels.')
    parser.add_argument('-feats_dir_train', default='data/{}/{}/train'.format(task,flevel),
                        help='Path to the folder containing the TRAIN features.')
    parser.add_argument('-feats_dir_dev', default='data/{}/{}/dev'.format(task, flevel),
                        help='Path to the folder containing the DEV features.')
    parser.add_argument('-feats_dir_test', default='data/{}/{}/test'.format(task, flevel),
                        help='Path to the folder containing the TEST features.')
    parser.add_argument('-model_out_dir', default='data/{}/{}'.format(task, flevel),
                        help='Path to the folder containing the model.')

    parser.add_argument('-input_dim', action="store_true", default=40)
    parser.add_argument('-net_output', action="store_true", default=1)
    parser.add_argument('-batch_size', action="store_true", default=32)
    parser.add_argument('-num_epochs', action="store_true", default=100)

    parser.add_argument('-training_mode', default='init', help='(init) Train from scratch, (resume) Resume training, '
                                                               '(finetune) Finetune a pretrained model')
    parser.add_argument('-model_type', default='TransformerPrime', help='DNN Model class. Check dnn_models.py')
    parser.add_argument('-base_LR', default=1e-3, type=float, help='Initial LR')
    parser.add_argument('-max_LR', default=2e-3, type=float, help='Maximum LR')

    return parser


def set_parameter_requires_grad(model, grad):
    for param in model.parameters():
        param.requires_grad = grad


# TORCHVISION PRE-TRAINED MODELS #
def initialize_model(model_name, num_classes, grad, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, grad)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # adapting number of channels from 3 (originally) to 1 (for audio)
        model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        input_size = 224

    if model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, grad)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # adapting number of channels from 3 (originally) to 1 (for audio)
        model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, grad)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, grad)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, grad)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, grad)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, grad)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
