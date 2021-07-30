import os
from collections import defaultdict
from time import time

import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms

from CustomDataset import DementiaDataset

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold


# Create the classifier that uses BERT
class DementiaDiscriminator(nn.Module):

    def __init__(self, n_classes):
        super(DementiaDiscriminator, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_samples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for i in data_loader:
        input_ids = i["input_ids"].to(device)
        attention_mask = i["attention_mask"].to(device)
        labels = i["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions / n_samples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_samples):
  model = model.eval()
  losses = []

  correct_predictions = 0

  with torch.no_grad():
    for i in data_loader:
        input_ids = i["input_ids"].to(device)
        attention_mask = i["attention_mask"].to(device)
        labels = i["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

  return correct_predictions / n_samples, np.mean(losses)


if __name__ == '__main__':

    # tokenizer = BertTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')
    # model = BertModel.from_pretrained("SZTAKI-HLT/hubert-base-cc")

    pre_trained_model_name = 'distilbert-base-multilingual-cased'
    # pre_trained_model = 'bert-base-cased'
    # pre_trained_model = 'SZTAKI-HLT/hubert-base-cc'

    tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
    # model = BertModel.from_pretrained(pre_trained_model, return_dict=False)

    class_names = [1, 2, 3]
    # class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Load data
    dataset = DementiaDataset(transcriptions_dir='../data/text/dementia94B',
                              labels_dir='../data/text/dementia94B/labels', tokenizer=tokenizer,
                              max_len=250, tokens_to_exclude=['[SIL]', '[EE]', '[MM]', '[OOO]',
                                                              '[BREATH]', '[AAA]', '[PAU]'])

    # train_loader = DataLoader(dataset=dataset, batch_size=8, num_workers=0)

    # Splitting data for CV
    kfold = KFold(n_splits=10, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = DataLoader(dataset=dataset, batch_size=10, num_workers=0, sampler=train_subsampler)
        test_loader = DataLoader(dataset=dataset, batch_size=10, num_workers=0, sampler=test_subsampler)

        best_accuracy = 0

        # Instantiate the model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DementiaDiscriminator(len(class_names))
        model = model.to(device)

        # train the model
        EPOCHS = 50
        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        loss_fn = nn.CrossEntropyLoss().to(device)

        for epoch in range(EPOCHS):
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)
            train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device,
                                                scheduler, len(train_loader))
            print(f'Train loss {train_loss} accuracy {train_acc}')
