"""
Created by José Vicente Egas López
on 2021. 02. 01. 13 00

"""
import argparse
import os
from datetime import datetime
from prettytable import PrettyTable

import kaldi_python_io
import numpy as np

import dnn_models

import torch


def prepare_model(args):
    if args.training_mode == 'init':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print('Initializing Model...')
        step = 0
        net = eval('dnn_models.{0}({1}, {2}, p_dropout=0.4)'.format(args.model_type, args.input_dim, args.num_classes))
        print(net)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.base_LR)

        event_ID = datetime.now().strftime('%Y%m-%d%H-%M%S')
        save_dir = '{}/models/modelType_{}_event_{}'.format(args.model_out_dir, args.model_type, event_ID)
        os.makedirs(save_dir)

        return net, optimizer, step, save_dir


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


def get_train_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-feat_type', type=str, default='melspecT', help="Type of the frame-level features to load or"
                                                                     "extract. Available types: mfcc, fbanks, spec,"
                                                                     "melspecT")
    parser.add_argument('-config_file', default='conf/mel_spectrogram.ini', help='Path to the config (ini) file.')
    parser.add_argument('-deltas', default=0, type=int, help="Compute delta coefficients of a tensor. '1' for "
                                                             "first order derivative, '2' for second order. "
                                                             "None for not using deltas. Default: None.")

    parser.add_argument('-online', default=True, required=False, help='If True, features are computed on the fly. '
                                                                      'If False, features are loaded from disk from '
                                                                      'specified input of -. Default: True')
    parser.add_argument('-labels', default='data/sleepiness/labels/labels_orig.csv', required=False, help='Path to the file '
                                                                        'containing the labels.')
    parser.add_argument('-feats_dir_train', default='data/sleepiness/mfcc/train',
                        help='Path to the folder containing the features.')
    parser.add_argument('-feats_dir_dev', default='data/sleepiness/mfcc/dev',
                        help='Path to the folder containing the features.')

    parser.add_argument('-model_out_dir', default='data/sleepiness/mfcc',
                        help='Path to the folder containing the features.')

    parser.add_argument('-input_dim', action="store_true", default=40)
    parser.add_argument('-num_classes', action="store_true", default=10)
    parser.add_argument('-batch_size', action="store_true", default=32)
    parser.add_argument('-num_epochs', action="store_true", default=20)

    parser.add_argument('-training_mode', default='init',
                        help='(init) Train from scratch, (resume) Resume training, (finetune) Finetune a pretrained model')
    parser.add_argument('-model_type', default='TransformerPrime', help='Model class. Check dnn_models.py')
    parser.add_argument('-base_LR', default=1e-3, type=float, help='Initial LR')
    parser.add_argument('-max_LR', default=2e-3, type=float, help='Maximum LR')


    return parser

