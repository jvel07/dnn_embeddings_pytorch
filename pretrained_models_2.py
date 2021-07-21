"""
Created by José Vicente Egas López
on 2021. 03. 03. 16 54

"""
import copy
import time

from sklearn.metrics import recall_score
from torch.utils.data import DataLoader
from torchvision.models import resnet101
import torch
import torch.nn
import torch.optim as optim

import numpy as np

import train_utils
from CustomDataset import CustomAudioDataset
from feature_extraction import get_feats

# task (name of the dataset)
task = 'mask'
# in and out dirs
corpora_dir = '/media/jose/hk-data/PycharmProjects/the_speech/audio/'
out_dir = 'data/' + task
task_audio_dir = corpora_dir + task + '/'

# Get params
parser = train_utils.get_train_params(task, 'mfcc')
args = parser.parse_args()
if not args.online and (args.feat_dir_train is None or args.feat_dir_dev is None):
    parser.error("When -online=False, please specify -feat_dir_train adn -feat_dir_dev.")

# Loading the data
# Loading the data
train_set = CustomAudioDataset(file_labels=args.labels, audio_dir=task_audio_dir, online=args.online,
                               feats_fir=args.feats_dir_train, max_length_sec=25,
                               calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir,
                                                                 feat_type=args.feat_type,
                                                                 deltas=args.deltas, config_file=args.config_file)
                               )
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, drop_last=False, pin_memory=True)

dev_set = CustomAudioDataset(file_labels=args.labels, audio_dir=task_audio_dir, online=args.online,
                             feats_fir=args.feats_dir_dev, max_length_sec=25,
                             calc_flevel=get_feats.FLevelFeatsTorch(save=True, out_dir=out_dir,
                                                               feat_type=args.feat_type,
                                                               deltas=args.deltas, config_file=args.config_file)
                             )
dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False, pin_memory=True)


# Train model
def train_model(model, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, sample_batched in enumerate(eval('{}_loader'.format(phase))):
                inputs = sample_batched['feature'].to(device).unsqueeze(1)
                # x_train = torch.transpose(x_train, 1, -1)#.unsqueeze(1)
                labels = sample_batched['label'].to(device)
                # torch.cuda.empty_cache()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        print(outputs.shape)
                        print(labels.shape)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            # uar = recall_score(truths_list, preds_list, average='macro')
            epoch_loss = running_loss / len(eval('{}_loader.dataset'.format(phase)))
            epoch_acc = running_corrects.double() / len(eval('{}_loader.dataset'.format(phase)))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'dev' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'dev':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# Finetuning the convnet: Instead of random initializaion, we initialize the network with a pretrained network,
# like the one that is trained on imagenet 1000 dataset. Rest of the training looks as usual.
# ConvNet as fixed feature extractor: Here, we will freeze the weights for all of the network except that of the final
# fully connected layer. This last fully connected layer is replaced with a new one with random weights and only this
# layer is trained.
grad = True  # True: finetune the whole model; False: only update the reshaped layer params  (use it as feat. extractor)
model_name = 'resnet101'
# Initialize the model
net, input_size = train_utils.initialize_model(args.model_type, args.num_classes, grad, use_pretrained=True)

# Detect GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = net.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
as_feat_extract = False
params_to_update = model_ft.parameters()
print("Params to learn:")
if as_feat_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Optimizer
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = torch.nn.BCEWithLogitsLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, criterion, optimizer, num_epochs=args.num_epochs,
                             is_inception=(model_name == "inception"))
