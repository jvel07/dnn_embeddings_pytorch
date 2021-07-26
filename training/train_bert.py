from CustomDataset import DementiaDataset

from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# tokenizer = BertTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')
# model = BertModel.from_pretrained("SZTAKI-HLT/hubert-base-cc")

tokenizer = BertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = BertModel.from_pretrained("distilbert-base-multilingual-cased")

# Load data
train_set = DementiaDataset(transcriptions_dir='../data/text/dementia94B',
                            labels_dir='../data/text/dementia94B/labels', tokenizer=tokenizer,
                            max_len=128, tokens_to_exclude=['[SIL]', '[EE]', '[MM]', '[OOO]',
                                                            '[BREATH]', '[AAA]', '[PAU]'])

train_loader = DataLoader(dataset=train_set, batch_size=8, num_workers=0)
