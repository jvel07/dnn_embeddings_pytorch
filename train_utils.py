"""
Created by José Vicente Egas López
on 2021. 02. 01. 13 00

"""
import argparse
import os
from datetime import datetime
import dnn_models

import torch


def prepare_model(args):
    if args.trainingMode == 'init':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print('Initializing Model..')
        step = 0
        net = eval('dnn_models.{0}({1}, {2}, p_dropout=0)'.format(args.modelType, args.featDim, args.nClasses))
        print(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.baseLR)

        net.to(device)
        eventID = datetime.now().strftime('%Y%m-%d%H-%M%S')
        saveDir = './models/modelType_{}_event_{}'.format(args.modelType, eventID)
        os.makedirs(saveDir)

        return net, optimizer, step, saveDir


def getParams():
    parser = argparse.ArgumentParser()

    # PyTorch distributed run
    parser.add_argument("--local_rank", type=int, default=0)

    # General Parameters
    parser.add_argument('-modelType', default='xvecTDNN', help='Model class. Check models.py')
    parser.add_argument('-featDim', default=40, type=int, help='Frame-level feature dimension')
    parser.add_argument('-trainingMode', default='init',
        help='(init) Train from scratch, (resume) Resume training, (finetune) Finetune a pretrained model')
    parser.add_argument('-resumeModelDir', default=None, help='Path containing training checkpoints')
    parser.add_argument('-featDir', default=None, help='Directory with training archives')

    # Training Parameters - no more trainFullXvector = 0
    trainingArgs = parser.add_argument_group('General Training Parameters')
    trainingArgs.add_argument('-nClasses', default=10, type=int, help='Number of classes')
    # trainingArgs.add_argument('-numSpkrs', default=7323, type=int, help='Number of output labels')
    trainingArgs.add_argument('-logStepSize', default=200, type=int, help='Iterations per log')
    trainingArgs.add_argument('-batchSize', default=32, type=int, help='Batch size')
    # trainingArgs.add_argument('-numEgsPerArk', default=366150, type=int,
    #     help='Number of training examples per egs file')

    # Optimization Params
    optArgs = parser.add_argument_group('Optimization Parameters')
    optArgs.add_argument('-preFetchRatio', default=30, type=int, help='xbatchSize to fetch from dataloader')
    optArgs.add_argument('-optimMomentum', default=0.5, type=float, help='Optimizer momentum')
    optArgs.add_argument('-baseLR', default=1e-3, type=float, help='Initial LR')
    optArgs.add_argument('-maxLR', default=2e-3, type=float, help='Maximum LR')
    optArgs.add_argument('-numEpochs', default=2, type=int, help='Number of training epochs')
    optArgs.add_argument('-noiseEps', default=1e-5, type=float, help='Noise strength before pooling')
    optArgs.add_argument('-pDropMax', default=0.2, type=float, help='Maximum dropout probability')
    optArgs.add_argument('-stepFrac', default=0.5, type=float,
        help='Training iteration when dropout = pDropMax')

    # Metalearning params
    protoArgs = parser.add_argument_group('Protonet Parameters')
    protoArgs.add_argument('-preTrainedModelDir', default=None, help='Embedding model to initialize training')
    protoArgs.add_argument('-protoMinClasses', default=5, type=int, help='Minimum N-way')
    protoArgs.add_argument('-protoMaxClasses', default=35, type=int, help='Maximum N-way')
    protoArgs.add_argument('-protoEpisodesPerArk', default=25, type=int, help='Episodes per ark file')
    protoArgs.add_argument('-totalEpisodes', default=100, type=int, help='Number of training episodes')
    protoArgs.add_argument('-supportFrac', default=0.7, type=float, help='Fraction of samples as supports')

    return parser
