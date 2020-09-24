# This file is for converting data to the format of Senti4SD
# Created by happygirlzt

import pandas as pd
import numpy as np
import re

def convert_cr():   
    df=pd.read_csv('../data/cr-test-se.csv',header=None,encoding='utf_8')    
    df.columns=['Text','Polarity']
    df['Polarity']=df['Polarity'].replace({-1: 'negative', 1: 'positive', 0: 'neutral'})
    df.to_csv('../data/cr-test-sd.csv', index=False,encoding='utf_8')
convert_cr()

def convert_jira():
    df=pd.read_csv('../data/jira-test-se.csv',header=None)    
    df.columns=['Text','Polarity']
    df['Polarity']=df['Polarity'].replace({-1: 'negative', 1: 'positive'})
    df.to_csv('../data/jira-test-sd.csv', index=False)
#convert_jira()

def convert_so():
    for file_name in ['train','test']:
        df=pd.read_csv('../data/so-{}.csv'.format(file_name),usecols=['text','oracle'])
        df.columns=['Text','Polarity']        
        df['Polarity']=df['Polarity'].replace({-1: 'negative', 1: 'positive', 0: 'neutral'})
        
        df.to_csv('../data/so-{}-sd.csv'.format(file_name),index=False)
#convert_so()

def convert_api():    
    for file_name in ['train','test']:
        df=pd.read_csv('../data/api-{}.csv'.format(file_name), usecols=['sentence','label'])        
        df.columns=['Text','Polarity']        
        df['Polarity']=df['Polarity'].replace({-1: 'negative', 1: 'positive', 0: 'neutral'})        
        df.to_csv('../data/api-{}-sd.csv'.format(file_name),index=False)
#convert_api()

def convert_app():
    for file_name in ['train','test']:
        df=pd.read_csv('../data/app-{}.csv'.format(file_name), usecols=['sentence','oracle'])
        df.columns=['Text','Polarity']        
        df['Polarity']=df['Polarity'].replace({-1: 'negative', 1: 'positive', 0: 'neutral'})
        
        df.to_csv('../data/app-{}-sd.csv'.format(file_name), index=False)
#convert_app()

def convert_gh():
    df=pd.read_csv('../data/gh-test.csv', usecols=['Text','Polarity'])
    df.to_csv('../data/gh-test-sd.csv', index=False)
#convert_gh()