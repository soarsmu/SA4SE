# -*- coding: utf-8 -*-
from utils import *
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import argparse

#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertForSequenceClassification,BertTokenizer,'bert-base-cased'),
          (XLNetForSequenceClassification, XLNetTokenizer,'xlnet-base-cased'),
          (RobertaForSequenceClassification, RobertaTokenizer,'roberta-base'), 
          (AlbertForSequenceClassification, AlbertTokenizer,'albert-base-v1')
         ]

MODEL_NAMES = ['bert', 'xlnet', 'Roberta', 'albert']

seed_torch(42)

## Read model name
parser = argparse.ArgumentParser(description='Choose the models.')

parser.add_argument('-m', '--model_num', default=0, type=int, nargs='?',
                    help='Enter an integer... 0-BERT, 1-XLNet, 2-RoBERTa, 3-ALBERT; default: 0')


args = parser.parse_args()
m_num=args.model_num

cur_model=MODELS[m_num]
m_name=MODEL_NAMES[m_num]

train_df = pd.read_pickle(jira_train)
train_df['label']=train_df['label'].replace(-1, 0)
# Negative: 0, Positive: 1

tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

sentences=train_df.sentence.values
labels=train_df.label.values

# max_len = 0
# for sent in sentences:
#     input_ids=tokenizer.encode(sent, add_special_tokens=True)
#     max_len=max(max_len, len(input_ids))
# print('Max sentence length: ', max_len)

input_ids = []
attention_masks = []

for sent in sentences:

    encoded_dict = tokenizer.encode_plus(
                        str(sent), 
                        add_special_tokens = True, 
                        max_length = MAX_LEN,
                        pad_to_max_length = True,
                        return_attention_mask = True, 
                        return_tensors = 'pt'
                   )
     
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])


train_inputs = torch.cat(input_ids, dim=0)
train_masks = torch.cat(attention_masks, dim=0)
train_labels = torch.tensor(labels)

print('Training data {} {} {}'.format(train_inputs.shape, train_masks.shape, train_labels.shape))

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Train Model
model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

begin=time.time()
train_loss_set = []

for _ in trange(EPOCHS, desc="Epoch"): 

    model.train()

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    
    for step, batch in enumerate(train_dataloader):
    
        batch = tuple(t.to(device) for t in batch)
      
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(b_input_ids, token_type_ids=None, \
                        attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        train_loss_set.append(loss.item())    
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))

end=time.time()
print('Training used {} second'.format(end-begin))

begin=time.time()
test_df = pd.read_pickle(jira_test)
test_df['label']=test_df['label'].replace(-1, 0)

sentences=test_df.sentence.values
labels = test_df.label.values

input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                    str(sent), 
                    add_special_tokens = True, 
                    max_length = MAX_LEN,
                    pad_to_max_length = True,
                    return_attention_mask = True, 
                    return_tensors = 'pt'
                   )
     
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

prediction_inputs = torch.cat(input_ids,dim=0)
prediction_masks = torch.cat(attention_masks,dim=0)
prediction_labels = torch.tensor(labels)

prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

model.eval()
predictions,true_labels=[],[]

for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    predictions.append(logits)
    true_labels.append(label_ids)
    
end=time.time()
print('Prediction used {:.2f} seconds'.format(end-begin))

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

print("Accuracy of {} on Jira is: {}".format(m_name, accuracy_score(flat_true_labels,flat_predictions)))
print(classification_report(flat_true_labels, flat_predictions))