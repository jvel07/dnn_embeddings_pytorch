# DNN Embeddings
Extraction of DNN embeddings from utterances using PyTorch.

This is a project under constant development. The main objective of it is to use DNN to extract meaningful representations of frame-level audio features such as MFCCs, FBANKS, MelSpecs.
The requirements to run are:

- torch
- numpy
- pandas

```
import torch
import torch.nn as nn
import utils
from CustomDataset import CustomDataset
import get_feats
import numpy as np
```

The pipeline of the project is designed in a simple and straightforward manner. Just few steps and you can have frame-level features fed into a DNN to train.
For loading an audio dataset you would only need to do the following:

```
train_set = CustomDataset(file_labels='data/sleepiness/labels/labels.csv', audio_dir=task_audio_dir, 
                          name_set='train', online=True,
                          calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir, feat_type='mfcc', deltas=1, **params))
                          
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, drop_last=False)
```
The snippet above is intended to be used with a DataLoader which loads the utterances and calculates frame-level features on-the-fly (during training). It takes the labels and audio directories as parameters, it also takes the name of the sets (if available). You can specify what type of frame level feature you want to use. Choose between: 'mfcc', 'fbanks', 'melspec'; moreover, you can to compute their first and second derivatives if needed: use the 'deltas' parameter.

TO BE CONTINUED...
