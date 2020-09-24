# Created by happygirlzt
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from pathlib import Path
import re
import torchtext
import glob
from torchtext import data
from torchtext.data import Field


if torch.cuda.is_available():       
    device = torch.device("cuda")
    #print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    #print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

data_folder=Path('../data/')
model_folder=Path('../models/')
result_folder=Path('../result/')

#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertForSequenceClassification,BertTokenizer,'bert-base-cased'),
          (XLNetForSequenceClassification, XLNetTokenizer,'xlnet-base-cased'),
          (RobertaForSequenceClassification, RobertaTokenizer,'roberta-base'),
          (AlbertForSequenceClassification, AlbertTokenizer,'albert-base-v1')
         ]

MODEL_NAMES = ['bert', 'xlnet', 'Roberta', 'albert']

## Parameters setting
BATCH_SIZE=16
LEARNING_RATE=2e-5
MAX_SEQ_LENGTH=256
SEED=42
EPOCHS=4
EPS=1e-8
WEIGHT_DECAY=1e-5

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

seed_torch(SEED)


def get_dataloader(X_cur, y_cur, cur_model, is_train):
    input_ids, attention_masks = preprocessing_for_classifier_tensor(X_cur.values, cur_model)
    
    labels = torch.from_numpy(np.array(y_cur, dtype='int64'))

    cur_dataset = TensorDataset(input_ids, attention_masks, labels)
    
    cur_dataloader = DataLoader(
                cur_dataset,
                batch_size = BATCH_SIZE,
                shuffle=is_train)
    
    return cur_dataloader

def get_iterator(X_cur, y_cur, cur_model, is_train):
    input_ids, attention_masks = preprocessing_for_classifier_list(X_cur.values, cur_model)
    #print(f'type of input_ids: {type(input_ids)}')
    #print(f'type of input_ids[0]: {type(input_ids[0])}')
    #print(f'type of attention_masks: {type(attention_masks)}')
    #print(f'type of attention_masks[0]: {type(attention_masks[0])}')
    labels = torch.from_numpy(np.array(y_cur, dtype='int64'))
    
    INPUT_IDS=Field(sequential=False, use_vocab=False, batch_first=True)
    ATTENTION_MASKS=Field(sequential=False, use_vocab=False, batch_first=True)
    LABEL=Field(sequential=False, use_vocab=False, batch_first=True)
    
    fields=[
        ('INPUT_IDS', INPUT_IDS),
        ('ATTENTION_MASKS', ATTENTION_MASKS),
        ('LABEL', LABEL)
    ]
    examples=[]
    for i in range(len(labels)):
        examples.append(data.Example.fromlist([input_ids[i],
                                               attention_masks[i],
                                               labels[i]],
                                               fields))
    
    
    cur_dataset = torchtext.data.Dataset(examples, fields)
    cur_iterator = data.BucketIterator(cur_dataset, batch_size=BATCH_SIZE, device='cuda', shuffle=is_train)
    return cur_iterator

def preprocessing_for_classifier_tensor(sentences, cur_model):
    tokenizer=cur_model[1].from_pretrained(cur_model[2])
    input_ids=[]
    attention_masks=[]

    for sent in sentences:
        encoded_sent = tokenizer.encode_plus(
            str(sent),
            add_special_tokens=True, 
            max_length=MAX_SEQ_LENGTH,
            pad_to_max_length=True,
            return_tensors='pt',  # Return PyTorch tensor
            return_attention_mask=True
            )
        
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

def preprocessing_for_classifier_list(sentences, cur_model):
    tokenizer=cur_model[1].from_pretrained(cur_model[2])
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_sent = tokenizer.encode_plus(
            str(sent),
            add_special_tokens=True, 
            max_length=MAX_SEQ_LENGTH,
            pad_to_max_length=True,
            return_attention_mask=True
            )
        
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    return input_ids, attention_masks

 
def run_saved_model(prediction_dataloader, cur_model, p_name, m_name):
    model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
    model.cuda()
    # satd_classifier.load_state_dict(torch.load(data_folder/'{}-{}.bin'.format(p_name, m_name)))    
    # print('{}-{}.bin loaded'.format(p_name, m_name))
    
    name_pattern='/sa4se/models/best_{}_{}_*'.format(p_name, m_name)
    # print(type(glob.glob(name_pattern)))
    candidates=glob.glob(name_pattern)
    candidates.sort(reverse=True)
    file_name=candidates[0]
    
    model.load_state_dict(torch.load(file_name))
    print('{} loaded'.format(file_name))
    
    model.eval()
    predictions, true_labels = [], []
    
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
             outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        
        # will create a synchronization point
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    
    #print('Precision is {:.3f}'.format(precision_score(flat_true_labels, flat_predictions)))
    #print('Recall is {:.3f}'.format(recall_score(flat_true_labels, flat_predictions)))
    #print('F1-score is {:.3f}'.format(f1_score(flat_true_labels, flat_predictions)))    
    print(classification_report(flat_true_labels, flat_predictions))