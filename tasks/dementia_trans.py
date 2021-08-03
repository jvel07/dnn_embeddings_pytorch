import os
from collections import defaultdict

import numpy as np
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
from transformers import pipeline

from CustomDataset import DementiaDataset
from feature_extraction import get_feats

pre_trained_model_name = 'distilbert-base-multilingual-cased'
# pre_trained_model = 'bert-base-cased'
# pre_trained_model = 'SZTAKI-HLT/hubert-base-cc'

tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
model = BertModel.from_pretrained(pre_trained_model_name, return_dict=False)


class_names = [1, 2, 3]
# class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Load data
dataset = DementiaDataset(transcriptions_dir='../data/text/dementia94B/transcriptions',
                          labels_dir='../data/text/dementia94B/labels', tokenizer=tokenizer,
                          max_len=200, tokens_to_exclude=['[SIL]', '[EE]', '[MM]', '[OOO]',
                                                          '[BREATH]', '[AAA]', '[PAU]'],
                          calc_embeddings=get_feats.ExtractTransformersEmbeddings(model_name=pre_trained_model_name,
                                                                                  out_dir='../data/text/dementia94B/embeddings')
                          )

train_loader = DataLoader(dataset=dataset, batch_size=8, num_workers=0)

# Splitting data for CV
kfold = KFold(n_splits=10, shuffle=True)
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    train_loader = DataLoader(dataset=dataset, batch_size=10, num_workers=0, sampler=train_subsampler)
    test_loader = DataLoader(dataset=dataset, batch_size=10, num_workers=0, sampler=test_subsampler)

for i, sample in enumerate(train_loader):
    print(i, sample['transcription'], sample['label'])
