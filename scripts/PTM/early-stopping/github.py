# Created by happygirlzt
# -*- coding: utf-8 -*-
import sys
sys.path.append('/media/DATA/tingzhang-data/sa4se/scripts')

from utils import *
from sklearn.model_selection import train_test_split
import argparse
import pprint
import math
from transformers import AdamW
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage, Precision, Recall
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.contrib.handlers import ProgressBar

import logging
logging.basicConfig(level=logging.ERROR)

## Read model name and project name
parser = argparse.ArgumentParser(description='Choose the models.')

parser.add_argument('-m', '--model_num', default=0, type=int, nargs='?',
                    help='Enter an integer... 0-BERT, 1-XLNet, 2-RoBERTa, 3-ALBERT; default: 0')

parser.add_argument('-r', '--re_run', default=0, type=int, nargs='?',
                    help='Enter an integer... 0-re-run the saved model, 1-run new model; default: 0')

args = parser.parse_args()
#print(args.model_num)
#print(args.project_num)

m_num=args.model_num
rerun_flag=bool(args.re_run)
    
# Generate training, validation and test set
data_folder=Path('../data/')

cur_model=MODELS[m_num]
m_name=MODEL_NAMES[m_num]

print('Running model {} in Github'.format(m_name))

#### Read data
train_data=pd.read_pickle(data_folder/'gh-train.pkl')
train_data['label']=train_data['label'].replace({'positive':1, 'negative':2, 'neutral':0})

X_train=train_data['sentence']
y_train=train_data['label']

test_data=pd.read_pickle(data_folder/'gh-test.pkl')
test_data['label']=test_data['label'].replace({'positive':1, 'negative':2, 'neutral':0})

X_test=test_data['sentence']
y_test=test_data['label']
print('Read success!')

# pred_iterator=get_iterator(X_test, y_test, cur_model, False)

prediction_dataloader=get_dataloader(X_test, y_test, cur_model, False)

# print('Training set is {}\nValidation set is {}\nTest set is {}'.format(len(train_dataloader.dataset), len(validation_dataloader.dataset), len(prediction_dataloader.dataset)))

if rerun_flag:
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, 
                                                            y_train, 
                                                            test_size=0.05, 
                                                            random_state=SEED,
                                                            stratify=y_train)

    #train_dataloader=get_dataloader(X_train, y_train,cur_model,True)
    #validation_dataloader=get_dataloader(X_validation, y_validation,cur_model,False)
    
    train_iterator=get_iterator(X_train, y_train, cur_model, True)
    valid_iterator=get_iterator(X_validation, y_validation, cur_model, False)
    
    model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
    model.cuda()
    
    optimizer = AdamW(model.parameters(),
                      lr=LEARNING_RATE,
                      eps=EPS,
                      weight_decay=WEIGHT_DECAY)
    
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8) # 5e-5 * 0.8 = 4e-5
    
    def process_function(engine, batch):
        model.train()
        optimizer.zero_grad()
        
        b_input_ids = batch.INPUT_IDS
        b_input_mask = batch.ATTENTION_MASKS
        b_labels = batch.LABEL
            

        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs[0]
        logits = outputs[1]

        loss.backward()
        optimizer.step()
        #scheduler.step()
        return loss.item()

    def eval_function(engine, batch):
        model.eval()
        with torch.no_grad():
            b_input_ids = batch.INPUT_IDS
            b_input_mask = batch.ATTENTION_MASKS
            b_labels = batch.LABEL
            
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            #logits = outputs[0]
            y_pred=outputs[0]
            
            return y_pred, b_labels
    
    trainer = Engine(process_function)
    train_evaluator = Engine(eval_function)
    validation_evaluator = Engine(eval_function)
    
    #print('success!')
    #### Metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    
    def output_transform_fun(output):
        y_pred, y = output
        y_pred=y_pred.detach().cpu().numpy()
        y=y.to('cpu').numpy()
        y_pred=np.argmax(y_pred, axis=1).flatten()
        return torch.from_numpy(y_pred), torch.from_numpy(y)
    
    criterion = nn.CrossEntropyLoss()
    ### Training
    #Accuracy(output_transform=output_transform_fun).attach(train_evaluator, 'accuracy')
    Loss(criterion).attach(train_evaluator, 'cross-entropy')
    
    #precision = Precision(output_transform=output_transform_fun, average=False)
    #.detach().cpu().numpy()
    #recall = Recall(output_transform=output_transform_fun, average=False)
    #.detach().cpu().numpy()
    #F1 = (precision * recall * 2) / (precision + recall)

    #precision.attach(train_evaluator, 'precision')
    #recall.attach(train_evaluator, 'recall')
    #F1.attach(train_evaluator, 'F1')
    
    ### Validation    
    #Accuracy(output_transform=output_transform_fun).attach(validation_evaluator, 'accuracy')
    Loss(criterion).attach(validation_evaluator, 'cross-entropy')

    #precision.attach(validation_evaluator, 'precision')
    #recall.attach(validation_evaluator, 'recall')
    #F1.attach(validation_evaluator, 'F1')
    
    #### Progress Bar
    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(trainer, ['loss'])
    
    def score_function_loss(engine):
        val_loss = engine.state.metrics['cross-entropy']
        return -val_loss
    
    def score_function_f1(engine):
        val_f1 = engine.state.metrics['F1']
        if math.isnan(val_f1):
            return -9999
        return val_f1

    handler = EarlyStopping(patience=2, score_function=score_function_loss, trainer=trainer)
    
    validation_evaluator.add_event_handler(Events.COMPLETED, handler)
    
    def log_training_results(engine):
        train_evaluator.run(train_iterator)
        metrics = train_evaluator.state.metrics
        pbar.log_message(
        "Training Results - Epoch: {} \nMetrics\n{}"
        .format(engine.state.epoch, pprint.pformat(metrics)))
    
    def log_validation_results(engine):
        validation_evaluator.run(valid_iterator)
        metrics = validation_evaluator.state.metrics
        pbar.log_message(
        "Validation Results - Epoch: {} \nMetrics\n{}"
        .format(engine.state.epoch, pprint.pformat(metrics)))
        pbar.n = pbar.last_print_n = 0
        
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

    #### Checkpoint
    
    # to_save = {'{}_{}'.format(p_name, m_name): model,
    #           'optimizer': optimizer,
    #           'lr_scheduler': scheduler
    #           }
    
    to_save={'gh_{}'.format(m_name): model}
    
    cp_handler = Checkpoint(to_save,
                        DiskSaver('../models/',
                        create_dir=True, require_empty=False),
                        filename_prefix='best',
                        score_function=score_function_loss,
                        score_name='val_loss')

    validation_evaluator.add_event_handler(Events.COMPLETED, cp_handler)
    #trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), cp_handler)

    # checkpointer = ModelCheckpoint('../models/', '{}'.format(p_name), create_dir=True, save_as_state_dict=True, require_empty=False)
    
    # trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    trainer.run(train_iterator, max_epochs=4)
else:
    print('Runing saved model...')
    #run_on_test(cur_model, p_name, m_name, pred_iterator)
    run_saved_model(prediction_dataloader, cur_model, 'gh', m_name)