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


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_samples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for i, (data, label) in enumerate(data_loader):
        input_ids, labels = data.to(device), label.to(device)
        input_ids = torch.flatten(input_ids, start_dim=2)
        input_ids = input_ids.squeeze(dim=1).to(torch.long).to(device)
        # input_ids = torch.tensor(input_ids).to(torch.int64).to(device)
        output = model(input_ids)
        outputs = output[0].to(device)
        loss = loss_fn(outputs, labels)
        # print(loss)

        pred = outputs.max(1, keepdim=True)[1].to(device)
        correct_predictions += pred.eq(labels.view_as(pred)).sum().item()

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

  return correct_predictions.double() / n_samples, np.mean(losses)


if __name__ == '__main__':

    # pre_trained_model_name = 'distilbert-base-multilingual-cased'
    pre_trained_model_name = 'bert-base-cased'
    # pre_trained_model_name = 'SZTAKI-HLT/hubert-base-cc'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Load data
    # Prepare MNIST dataset by concatenating Train/Test part; we split later.
    tform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(16),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_set = MNIST(os.getcwd(), download=True, transform=tform_mnist, train=True)
    test_set = MNIST(os.getcwd(), download=True, transform=tform_mnist, train=False)
    dataset = ConcatDataset([train_set, test_set])

    # train_loader = DataLoader(dataset=train_set, batch_size=8, num_workers=0)
    # dev_loader = DataLoader(dataset=train_set, batch_size=8, num_workers=0)

    # Splitting data for CV
    kfold = KFold(n_splits=10, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(fold)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = DataLoader(dataset=dataset, batch_size=16, num_workers=2, sampler=train_subsampler)
        test_loader = DataLoader(dataset=dataset, batch_size=16, num_workers=2, sampler=test_subsampler)

        # Instantiate pre-trained BERT (BertForSequenceClassification)
        model = BertForSequenceClassification.from_pretrained(pre_trained_model_name, num_labels=len(class_names),
                                                              output_attentions=False, output_hidden_states=False, )

        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        EPOCHS = 20
        total_steps = len(train_loader) * EPOCHS
        print(total_steps, len(train_loader))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        loss_fn = nn.CrossEntropyLoss()

        history = defaultdict(list)

        best_accuracy = 0

        for epoch in range(EPOCHS):
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)
            train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device,
                                                scheduler, len(train_loader))
            print(f'Train loss {train_loss} accuracy {train_acc}')

