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
        net = eval('dnn_models.{0}({1}, {2}, p_dropout=0)'.format(args.model_type, args.input_dim, args.num_classes))
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

