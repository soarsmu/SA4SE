# The file is used to analyze the prediction performance of SentiStrength-SE
# Author: happygirlzt

import pandas as pd
import numpy as np

import re
from sklearn.metrics import classification_report, confusion_matrix

def get_confusion_matrix():
    pred=pd.read_csv('so-ss.csv', header=None)
    #print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    pos_list=[]
    neg_list=[]

    for sent in res_pd:
        cur_list=re.split(r'\t+', sent.rstrip('\t'))[1:]
        
        new_list=cur_list[0].split()
        
        pos_list.append(int(new_list[0]))
        neg_list.append(int(new_list[1]))

    total = [p + n for p, n in zip(pos_list, neg_list)]
    label=[]
    for score in total:
        if score>0:
            label.append(1)
        elif score==0:
            label.append(0)
        else:
            label.append(-1)

    pred_df=pd.Series(label,dtype='int32')

    test_df=pd.read_csv('so-test-se.csv',header=None)
    test_df.columns=['sentence','label']
    
    true_df=pd.Series(test_df['label'],dtype='int32')

    results = confusion_matrix(true_df,pred_df,labels=[1,0,-1])
    print(results)
    
#get_confusion_matrix()

def analyze_cr():
    # replace 'cr-ss.csv' with your prediction file name
    pred=pd.read_csv('cr-ss.csv',header=None)
    print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    pos_list=[]
    neg_list=[]

    for sent in res_pd:
        cur_list=re.split(r'\t+', sent.rstrip('\t'))[1:]
        #print(cur_list)
        
        new_list=cur_list[0].split()
        
        pos_list.append(int(new_list[0]))
        neg_list.append(int(new_list[1]))

    total = [p + n for p, n in zip(pos_list, neg_list)]
    label=[]
    for score in total:
        if score<0:
            label.append(-1)
        else:
            label.append(0)

    pred_df=pd.Series(label,dtype='int32')

    #print(pred_df)

    # read in true lables
    test_df=pd.read_csv('cr-test-se.csv',header=None)
    test_df.columns=['sentence','label']
    
    true_df=pd.Series(test_df['label'],dtype='int32')
    print(classification_report(true_df, pred_df))
    
    report=classification_report(true_df, pred_df, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('./SentiStrength-SE-cr1.csv')

analyze_cr()

def analyze_app():    
    pred=pd.read_csv('app-ss.csv',header=None)
    print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    pos_list=[]
    neg_list=[]

    for sent in res_pd:
        cur_list=re.split(r'\t+', sent.rstrip('\t'))[1:]
        #print(cur_list)
        new_list=cur_list[0].split()
        
        pos_list.append(int(new_list[0]))
        neg_list.append(int(new_list[1]))

    total = [p + n for p, n in zip(pos_list, neg_list)]
    label=[]
    for score in total:
        if score>0:
            label.append(1)
        elif score==0:
            label.append(0)
        else:
            label.append(-1)

    pred_df=pd.Series(label,dtype='int32')

    #print(pred_df)

    # read in true lables
    test_df=pd.read_csv('app-test-se.csv',header=None)
    test_df.columns=['sentence','label']
    
    true_df=pd.Series(test_df['label'],dtype='int32')
    print(classification_report(true_df, pred_df))
#analyze_app()

def analyze_gh():    
    pred=pd.read_csv('gh-ss.csv',header=None)
    print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    pos_list=[]
    neg_list=[]

    for sent in res_pd:
        cur_list=re.split(r'\t+', sent.rstrip('\t'))[1:]
        #print(cur_list)
        
        new_list=cur_list[0].split()
        
        pos_list.append(int(new_list[0]))
        neg_list.append(int(new_list[1]))

    total = [p + n for p, n in zip(pos_list, neg_list)]
    label=[]
    for score in total:
        if score>0:
            label.append(1)
        elif score==0:
            label.append(0)
        else:
            label.append(-1)

    pred_df=pd.Series(label,dtype='int32')

    #print(pred_df)

    # read in true lables
    test_df=pd.read_csv('gh-test-se.csv',header=None)
    test_df.columns=['sentence','label']
    
    test_df['label']=test_df['label'].replace({'positive':1, 'negative':-1, 'neutral':0})
    
    true_df=pd.Series(test_df['label'],dtype='int32')
    print(classification_report(true_df, pred_df))
    
#analyze_gh()

def analyze_jira():    
    pred=pd.read_csv('jira-ss.csv',header=None)
    print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    pos_list=[]
    neg_list=[]

    for sent in res_pd:
        cur_list=re.split(r'\t+', sent.rstrip('\t'))[1:]
        #print(cur_list)
        
        new_list=cur_list[0].split()
        
        pos_list.append(int(new_list[0]))
        neg_list.append(int(new_list[1]))

    total = [p + n for p, n in zip(pos_list, neg_list)]
    label=[]
    for score in total:
        if score>0:
            label.append(1)
        elif score==0:
            label.append(0)
        else:
            label.append(-1)

    pred_df=pd.Series(label,dtype='int32')

    #print(pred_df)

    # read in true lables
    test_df=pd.read_csv('jira-test-se.csv',header=None)
    test_df.columns=['sentence','label']
    
    true_df=pd.Series(test_df['label'],dtype='int32')
    print(classification_report(true_df, pred_df))
    
#analyze_jira()
def analyze_api():    
    pred=pd.read_csv('api-ss.csv',header=None)
    print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    pos_list=[]
    neg_list=[]

    for sent in res_pd:
        cur_list=re.split(r'\t+', sent.rstrip('\t'))[1:]
        #print(cur_list)
        
        new_list=cur_list[0].split()
        
        pos_list.append(int(new_list[0]))
        neg_list.append(int(new_list[1]))

    total = [p + n for p, n in zip(pos_list, neg_list)]
    label=[]
    for score in total:
        if score>0:
            label.append(1)
        elif score==0:
            label.append(0)
        else:
            label.append(-1)

    pred_df=pd.Series(label,dtype='int32')

    #print(pred_df)

    # read in true lables
    test_df=pd.read_csv('api-test-se.csv',header=None)
    test_df.columns=['sentence','label']
    
    true_df=pd.Series(test_df['label'],dtype='int32')
    print(classification_report(true_df, pred_df))

#analyze_api()

def analyze_so():    
    pred=pd.read_csv('so-ss.csv',header=None)
    print(pred.shape)
    pred.columns=['res']
    res_pd=pred['res']

    pos_list=[]
    neg_list=[]

    for sent in res_pd:
        cur_list=re.split(r'\t+', sent.rstrip('\t'))[1:]
        #print(cur_list)
        
        new_list=cur_list[0].split()
        
        pos_list.append(int(new_list[0]))
        neg_list.append(int(new_list[1]))

    total = [p + n for p, n in zip(pos_list, neg_list)]
    label=[]
    for score in total:
        if score>0:
            label.append(1)
        elif score==0:
            label.append(0)
        else:
            label.append(-1)

    pred_df=pd.Series(label,dtype='int32')

    #print(pred_df)

    # read in true lables
    test_df=pd.read_csv('so-test-se.csv',header=None)
    test_df.columns=['sentence','label']
    
    true_df=pd.Series(test_df['label'],dtype='int32')
    print(classification_report(true_df, pred_df))
#analyze_so()