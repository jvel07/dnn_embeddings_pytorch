# dnn_embeddings_pytorch
DNN embeddings extraction from utterances using PyTorch.

This is a project under constant development. The main objective of it is to use DNN to extract meaningful representations of frame-level audio features such as MFCCs, FBANKS, MelSpecs.
The requirements to run are:

- torch
- numpy
- pandas

The pipeline of the project is designed in a simple and straightforward manner. For loading an audio dataset you would only need to do the following:

```
train_set = CustomDataset(file_labels='data/sleepiness/labels/labels.csv', audio_dir=task_audio_dir, name_set='train', online=True,
                          calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir, feat_type='mfcc', deltas=1, **params))
```
The snippet above loads utterances and calculates frame-level features on-the-fly (during training). It takes the labels and audio directories as parameters, it also takes the name of the sets (if available). Once the wavs are loaded, you can specify what type of frame level feature you want to use. Choose between: 'mfcc', 'fbanks', 'melspec'.

TO BE CONTINUED...
