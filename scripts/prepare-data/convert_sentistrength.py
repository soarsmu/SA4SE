# This file is for convert data format to SentiStrength-SE format
# Created by happygirlzt

import pandas as pd
import numpy as np
import re
from pathlib import Path
data_folder=Path('YOUR_DATA_FOLDER')

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

def convert_jira_test():
    df=pd.read_pickle(jira_test)

    sents=[]

    for index, row in df.iterrows():
        text=row['sentence']
        text=''.join(text.split('\n'))
        sents.append(text)
    
    #print(len(sents))
    new_df=pd.DataFrame(sents,columns=['sentence'])

    df.update(new_df)
    df.to_csv('../data/jira-test-se.csv',header=None,index=False)

def convert_so_test():
    df=pd.read_pickle(so_test)
    sents=[]

    for index, row in df.iterrows():
        text=row['sentence']
        text=''.join(text.split('\n'))
        sents.append(text)
    
    #print(len(sents))
    new_df=pd.DataFrame(sents,columns=['sentence'])

    df.update(new_df)
    df.to_csv('../data/so-test-se.csv',header=None,index=False)
    
def convert_api_test():
    df=pd.read_pickle(api_test)
    sents=[]

    for index, row in df.iterrows():
        text=row['sentence']
        text=''.join(str(text).split('\n'))
        sents.append(text)
    
    #print(len(sents))
    new_df=pd.DataFrame(sents,columns=['sentence'])

    df.update(new_df)
    df.to_csv('../data/api-test-se.csv',header=None,index=False)
    
def convert_app_test():
    df=pd.read_pickle(app_test)
    sents=[]

    for index, row in df.iterrows():
        text=row['sentence']
        text=''.join(str(text).split('\n'))
        sents.append(text)
    
    #print(len(sents))
    new_df=pd.DataFrame(sents,columns=['sentence'])

    df.update(new_df)
    df.to_csv('../data/app-test-se.csv',header=None,index=False)
#convert_api_test() 

def convert_cr_test():
    df=pd.read_pickle(cr_test)

    sents=[]
    labels=[]
    for index, row in df.iterrows():
        text=row['sentence']
        text=''.join(text.split('\n'))
        sents.append(text)
        labels.append(row['label'])
        
    #print(len(sents))
    new_df=pd.DataFrame({'sentence': sents,'label': labels})
    
    sents=[]
    labels=[]
    for index, row in new_df.iterrows():
        text=row['sentence']
        text=''.join(text.split('\n'))
        sents.append(text)
        labels.append(row['label'])
    
    pd.DataFrame({'sentence': sents,'label': labels}).to_csv('../data/cr-test-se.csv',header=None,index=False)

#convert_so_test()
convert_cr_test()
#convert_app_test()

def convert_gh_test():
    df=pd.read_pickle(gh_test)
    sents=[]

    for index, row in df.iterrows():
        text=row['sentence']
        text=''.join(text.split('\n'))
        sents.append(text)
    
    #print(len(sents))
    new_df=pd.DataFrame(sents,columns=['sentence'])

    df.update(new_df)
    df.to_csv('../data/gh-test-se.csv',header=None,index=False)
#convert_gh_test()