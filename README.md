# DNN Embeddings for Speech and Audio Processing
Extraction of DNN embeddings from utterances using PyTorch.

This is a project under constant development, there may be parts that have to be concluded or enhanced yet. The main objective of it is to use DNNs to extract meaningful representations of frame-level audio features such as MFCCs, FBANKS, MelSpecs. We will try some types of DNNs for this purpose. For example, we implemented the DNN architecture to extract x-vector embeddings, which is based on this [paper](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf).

X-vectors are the current state-of-the-art for speaker verification, here we use these representations for medical speech processing and computational paralinguistic tasks.

Also, we will try different types of CNNs, and attention/transformer networks too.
Here we will describe how to train a DNN that can be employed to extract x-vectors.
The required libraries are:

- torch
- numpy
- pandas

The pipeline of the project is designed in a simple and straightforward manner. Just few steps and you can have frame-level features fed into a DNN to train it. The DNN structure is based on the above mentioned [paper](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf). The network is utilized to extract the so-called x-vectors embeddings from the utterances. 

*NOTE: 
Here, apart from getting x-vector embeddings, we will also attempt to get different types of **DNN embeddings** by experimenting with Transformers, and CNNs as well. MOREOVER, this project is designed for Speech Analysis, Medical Speech Processing and Computational Paralinguistics, specially for **Feature Extraction** phases. This project is **NOT** designed/intended for Speech or Speaker Recognition (although the methodologies/techniques used are based on them).*

Let's export the libraries we will need:

```
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import utils
import get_feats
from CustomDataset import CustomDataset
```

Now let's load the dataset, you would only need to do the following:

```
train_set = CustomDataset(file_labels='data/sleepiness/labels/labels.csv', audio_dir=task_audio_dir, 
                          name_set='train', online=True,
                          calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir, feat_type='mfcc', deltas=1, **params))

```

The snippet above is intended to be used with a `DataLoader`, which loads the utterances and calculates frame-level features on-the-fly (during training). `CustomDataset` takes the labels and audio directories as parameters, it also takes the name of the sets (if available). You can specify what type of frame level feature you want to use. Choose between: 'mfcc', 'fbanks', 'melspec'; moreover, you can to compute their first and second derivatives if needed: use the 'deltas' parameter. Now we can instantiate a `DataLoader` class (from torch.utils.data) and pass our `CustomDataset` as the (first) dataset parameter, the rest of the parameters depend on your criteria and needs:

```                       
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, drop_last=False)
```

Now we can proceed to prepare the model:

```
# Set the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# prepare the model
net, optimizer, step, saveDir = train_utils.prepare_model(args)
criterion = nn.CrossEntropyLoss()
# LR scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=args.maxLR,
                                                       cycle_momentum=False,
                                                       div_factor=5,
                                                       final_div_factor=1e+3,
                                                       total_steps=args.num_epochs * len(train_loader),
                                                       pct_start=0.15)
```

TO BE CONTINUED...
