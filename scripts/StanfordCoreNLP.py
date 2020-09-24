#!/usr/bin/env python
# Test data on Stanford CoreNLP
# Author: happygirlzt
# coding: utf-8
from sklearn.metrics import classification_report
from pycorenlp import StanfordCoreNLP
import pandas as pd
import time

from pathlib import Path
data_folder=Path('/sa4se/data') # your data folder

api_train=data_folder/'api-train.pkl'
api_test=data_folder/'api-test.pkl'

gh_train=data_folder/'gh-train.pkl'
gh_test=data_folder/'gh-test.pkl'

jira_train=data_folder/'jira-train.pkl'
jira_test=data_folder/'jira-test.pkl'

so_train=data_folder/'so-train.pkl'
so_test=data_folder/'so-test.pkl'

app_train=data_folder/'app-train.pkl'
app_test=data_folder/'app-test.pkl'

cr_train=data_folder/'cr-train.pkl'
cr_test=data_folder/'cr-test.pkl'

nlp = StanfordCoreNLP('http://localhost:9000')

def get_predictions(test_df):
    print('total length is {}'.format(test_df.shape[0]))
    predictions=[]

    for index, row in test_df.iterrows():
        sent=row['sentence']
        #print(sent)
        try:
            res = nlp.annotate(sent,
                            properties={
                                'annotators': 'sentiment',
                                'outputFormat': 'json',
                                'timeout': 5000000000000,
                            })
        except:
            # print(sent)
            predictions.append('Neutral')
            continue
        
        #print(type(res['sentences']))
        #return predictions
        
        try:
            # one row has many sentences
            if len(res['sentences']) > 1:
                total=0
                num=len(res['sentences'])
                
                for s in res['sentences']: 
                    # print(s['sentiment'])
                    # predictions.append(s['sentiment'])
                    total+=int(s['sentimentValue'])
                    
                if total / num == 2:
                    predictions.append('Neutral')
                elif total / num < 2:
                    predictions.append('Negative')
                else:
                    predictions.append('Positive')
            else:
                # one row only has one sentence
                for s in res['sentences']: 
                    #print(s['sentiment'])
                    predictions.append(s['sentiment'])
        except:
            predictions.append('Neutral')
            continue
    return predictions

def get_pred_df(cur_pred):
    pred_df=pd.DataFrame(cur_pred, columns=['Polarity'])
    pred_df['Polarity']=pred_df['Polarity'].replace({
        'Neutral':0,
        'Negative':-1,
        'Positive':1,
        'Verynegative':-1,
        'Verypositive':1})

    pred_df['Polarity']=pred_df['Polarity'].astype(int)
    return pred_df

def test_api():
    begin=time.time()
    # API reviews
    test_df=pd.read_pickle(api_test)
    cur_pred=get_predictions(test_df)
    
    end=time.time()
    print('Predict API used {:.2f} seconds'.format(end-begin))
    
    pred_df=get_pred_df(cur_pred)
    print(classification_report(test_df['label'], pred_df['Polarity']))

def test_gh():
    begin=time.time()
    # GitHub
    test_df=pd.read_pickle(gh_test)
    cur_pred=get_predictions(test_df)
    end=time.time()
    print('Predict GitHub used {:.2f} seconds'.format(end-begin))
    #len(predictions)
    pred_df=get_pred_df(cur_pred)

    test_df['label']=test_df['label'].replace({
        'neutral':0,
        'positive':1,
        'negative':-1})

    print(classification_report(test_df['label'], pred_df['Polarity']))

# APP reviews
def test_app():
    begin=time.time()
    test_df=pd.read_pickle(app_test)
    
    cur_pred=get_predictions(test_df)
    
    end=time.time()
    print('Predict APP used {:.2f} seconds'.format(end-begin))
    pred_df=get_pred_df(cur_pred)

    print(classification_report(test_df['label'], pred_df['Polarity']))

# SO
def test_so():
    begin=time.time()
    test_df=pd.read_pickle(so_test)
    
    cur_pred=get_predictions(test_df)
    end=time.time()
    print('Predict StackOverflow used {:.2f} seconds'.format(end-begin))
    
    pred_df=get_pred_df(cur_pred)
    print(classification_report(test_df['label'], pred_df['Polarity']))


# Jira
def test_jira():
    begin=time.time()
    test_df=pd.read_pickle(jira_test)
    cur_pred=get_predictions(test_df)
    
    end=time.time()
    print('Predict Jira used {:.2f} seconds'.format(end-begin))

    pred_df=get_pred_df(cur_pred)

    print(classification_report(test_df['label'], pred_df['Polarity']))


# CR
def test_cr():
    begin=time.time()
    test_df=pd.read_pickle(cr_test)
    cur_pred=get_predictions(test_df)
    
    end=time.time()
    print('Predict Code Reviews used {:.2f} seconds'.format(end-begin))
    
    pred_df=get_pred_df(cur_pred)
    print(classification_report(test_df['label'], pred_df['Polarity']))
    
#test_gh()
#test_api()
#test_app()
#test_so()
#test_jira()
#test_cr()