# This file is to used to predict the performance of Senti4SD
# Author: happygirlzt
import pandas as pd
import numpy as np

import re
from sklearn.metrics import classification_report,confusion_matrix

def get_confusion_matrix():
    pred=pd.read_csv('./predictions/so-predictions.csv',usecols=['PREDICTED'])
    
    #print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    test_df=pd.read_csv('so-test-sd.csv',usecols=['Text','Polarity'])
    
    true_df=pd.Series(test_df['Polarity'],dtype='int32')
    results=confusion_matrix(true_df,res_pd, labels=['positive','neutral','negative'])
    print(results)
    
#get_confusion_matrix()

def analyze_cr():
    # Replace './predictions/cr-predictions.csv' with your predicted file name
    pred=pd.read_csv('./predictions/cr-predictions.csv',usecols=['PREDICTED'])
    pred['PREDICTED']=pred['PREDICTED'].replace({'positive':'neutral'})
    
    print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    # read in true lables
    test_df=pd.read_csv('cr-test-sd.csv', usecols=['Text','Polarity'])
    
    true_df=pd.Series(test_df['Polarity'], dtype='int32')
    print(classification_report(true_df, pred))
#analyze_cr()

def analyze_app():    
    pred=pd.read_csv('./predictions/app-predictions.csv',usecols=['PREDICTED'])
    
    print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    # read in true lables
    test_df=pd.read_csv('app-test-sd.csv',usecols=['Text','Polarity'])
    
    true_df=pd.Series(test_df['Polarity'],dtype='int32')
    print(classification_report(true_df, pred))
#analyze_app()

def analyze_gh():    
    pred=pd.read_csv('./predictions/gh-predictions.csv',usecols=['PREDICTED'])
    
    print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    # read in true lables
    test_df=pd.read_csv('gh-test-sd.csv',usecols=['Text','Polarity'])
    
    true_df=pd.Series(test_df['Polarity'],dtype='int32')
    print(classification_report(true_df, pred))
#analyze_gh()

def analyze_jira():    
    pred=pd.read_csv('./predictions/jira-predictions.csv',usecols=['PREDICTED'])
    
    print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    # read in true lables
    test_df=pd.read_csv('jira-test-sd.csv',usecols=['Text','Polarity'])
    
    true_df=pd.Series(test_df['Polarity'],dtype='int32')
    print(classification_report(true_df, pred))
#analyze_jira()

def analyze_api():    
    pred=pd.read_csv('./predictions/api-predictions.csv',usecols=['PREDICTED'])
    
    print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    test_df=pd.read_csv('api-test-sd.csv',usecols=['Text','Polarity'])
    
    true_df=pd.Series(test_df['Polarity'],dtype='int32')
    print(classification_report(true_df, pred))    
#analyze_api()

def analyze_so():    
    pred=pd.read_csv('./predictions/so-predictions.csv',usecols=['PREDICTED'])
    
    print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    test_df=pd.read_csv('so-test-sd.csv',usecols=['Text','Polarity'])
    
    true_df=pd.Series(test_df['Polarity'],dtype='int32')
    print(classification_report(true_df, pred))
#analyze_so()