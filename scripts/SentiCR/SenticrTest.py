# Created by happygirlzt

from SentiCR import SentiCR

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import time

sentiment_analyzer=SentiCR()

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

def predict_jira():
    begin=time.time()
    df=pd.read_pickle(jira_test)

    df['label']=df['label'].replace(-1, 0)

    sentences=df['sentence']
    y_test=df['label']
    
    pred=[]
    for sent in sentences:
        score=sentiment_analyzer.get_sentiment_polarity(sent)
        pred.append(score)
    end=time.time()
    print('Prediction used {:.2f} seconds'.format(end-begin))
    
    y_pred=pd.DataFrame(pred, columns=['pred_label'])
    print(classification_report(y_test, y_pred))
    # report = classification_report(y_test, y_pred, output_dict=True)
    # df = pd.DataFrame(report).transpose()
    # df.to_csv('./SentiCR_jira.csv')

def predict_so():
    begin=time.time()
    df=pd.read_pickle(so_test)

    df['label']=df['label'].replace(-1, 2)

    sentences=df['sentence']
    y_test=df['label']
    
    pred=[]
    for sent in sentences:
        score=sentiment_analyzer.get_sentiment_polarity(sent)
        pred.append(score)
    
    end=time.time()
    print('Prediction used {:.2f} seconds'.format(end-begin))
    y_pred=pd.DataFrame(pred, columns=['pred_label'])
    
    print(classification_report(y_test, y_pred))
    #results = confusion_matrix(y_test, y_pred, labels=[1,0,2])
    #print(results)
    #report = classification_report(y_test, y_pred, output_dict=True)
    #df = pd.DataFrame(report).transpose()
    #df.to_csv('./SentiCR_so.csv')

def predict_gh():
    begin=time.time()
    df=pd.read_pickle(gh_test)

    sentences=df['sentence']
    y_test=df['label']
    
    pred=[]
    for sent in sentences:
        score=sentiment_analyzer.get_sentiment_polarity(sent)
        pred.append(score)
    
    end=time.time()
    print('Prediction used {:.2f} seconds'.format(end-begin))
    y_pred=pd.DataFrame(pred, columns=['pred_label'])
    
    # new_df=pd.DataFrame(columns=['Text', 'SentiCR_predicted'])
    # new_df['Text'] = sentences.copy
    # new_df['SentiCR_predicted'] = y_pred.copy
    
    # new_df.to_csv('./senticr_preditied.csv', header=True)
    
    print(classification_report(y_test, y_pred))
    # report = classification_report(y_test, y_pred, output_dict=True)
    # df = pd.DataFrame(report).transpose()
    # df.to_csv('./SentiCR_gh.csv')

def predict_app():
    begin=time.time()
    df=pd.read_pickle(app_test)

    df['label']=df['label'].replace(-1,2)

    sentences=df['sentence']
    y_test=df['label']
    
    print(sentences.shape[0]==y_test.shape[0])
    pred=[]
    for sent in sentences:
        score=sentiment_analyzer.get_sentiment_polarity(sent)
        pred.append(score)
    
    end=time.time()
    print('Prediction used {:.2f} seconds'.format(end-begin))
    y_pred=pd.DataFrame(pred, columns=['pred_label'])
    print(classification_report(y_test, y_pred))
    # report = classification_report(y_test, y_pred, output_dict=True)
    # df = pd.DataFrame(report).transpose()
    # df.to_csv('./SentiCR_app.csv')
    
def predict_cr():
    begin=time.time()
    df=pd.read_pickle(cr_test)
    df['label']=df['label'].replace(-1,1)

    sentences=df['sentence']
    y_test=df['label']
    
    pred=[]
    for sent in sentences:
        score=sentiment_analyzer.get_sentiment_polarity(sent)
        pred.append(score)
        
    end=time.time()
    print('Prediction used {:.2f} seconds'.format(end-begin))
    y_pred=pd.DataFrame(pred, columns=['pred_label'])
    print(classification_report(y_test, y_pred))
    # report = classification_report(y_test, y_pred, output_dict=True)
    # df = pd.DataFrame(report).transpose()
    # df.to_csv('./SentiCR_cr1.csv')

def predict_api():
    begin=time.time()
    df=pd.read_pickle(api_test)
    df['label']=df['label'].replace(-1,2)

    sentences=df['sentence']
    y_test=df['label']
    
    pred=[]
    for sent in sentences:
        score=sentiment_analyzer.get_sentiment_polarity(sent)
        pred.append(score)
    
    end=time.time()
    print('Prediction used {:.2f} seconds'.format(end-begin))
    y_pred=pd.DataFrame(pred, columns=['pred_label'])
    print(classification_report(y_test, y_pred))
    #report = classification_report(y_test, y_pred, output_dict=True)
    #df = pd.DataFrame(report).transpose()
    #df.to_csv('./SentiCR_api.csv')

#predict_jira()
#predict_api()
#predict_gh()
predict_so()
#predict_cr()
#predict_app()