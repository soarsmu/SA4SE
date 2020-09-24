# The file is used to analyze the performance of SentiStrength
# Author: happygirlzt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 

#lol = list(csv.reader(open('text.txt', 'rb'), delimiter='\t'))

def get_confusion_matrix():
    df=pd.read_csv('so-test+results.txt', sep='\t', index_col=False, header=None)
    #print(df.head())
    df.columns=['sent','pos','neg']
    #print(df.shape)

    result=[]
    total_lines=df.shape[0]
    for i in range(total_lines):
        cur_sum=int(df.iloc[i].pos)+int(df.iloc[i].neg)
        if cur_sum > 0:
            result.append(1)
        elif cur_sum == 0:
            result.append(0)
        else:
            result.append(-1)
            
    y_pred=pd.DataFrame(result)

    y_true=pd.read_csv('so-test-se.csv', header=None,usecols=[1])
    
    results = confusion_matrix(y_true,y_pred,labels=[1,0,-1])
    print(results)
    
#get_confusion_matrix()

def analyze(file_name):
    #replace '{}-test+results.txt' with your prediction file name
    df=pd.read_csv('{}-test+results.txt'.format(file_name), sep='\t', index_col=False, header=None)
    print(df.head())
    df.columns=['sent','pos','neg']
    print(df.shape)

    result=[]
    total_lines=df.shape[0]
    for i in range(total_lines):
        cur_sum=int(df.iloc[i].pos)+int(df.iloc[i].neg)
        if cur_sum > 0:
            result.append(1)
        elif cur_sum == 0:
            result.append(0)
        else:
            result.append(-1)
            
    y_pred=pd.DataFrame(result)

    y_true=pd.read_csv('{}-test-se.csv'.format(file_name), header=None,usecols=[1])

    print(classification_report(y_true, y_pred))
    
def analyze_gh():
    df=pd.read_csv('gh-test+results.txt', sep='\t', index_col=False, header=None)
    print(df.head())
    df.columns=['sent','pos','neg']
    print(df.shape)

    result=[]
    total_lines=df.shape[0]
    for i in range(total_lines):
        cur_sum=int(df.iloc[i].pos)+int(df.iloc[i].neg)
        if cur_sum > 0:
            result.append(1)
        elif cur_sum == 0:
            result.append(0)
        else:
            result.append(-1)
            
    y_pred=pd.DataFrame(result)

    y_true=pd.read_csv('gh-test-se.csv', header=None, usecols=[1])
    y_true=y_true.replace({'positive':1, 'negative':-1, 'neutral':0})

    print(classification_report(y_true,y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('./SentiStrength-gh.csv')
#analyze('api')
#analyze('cr')
#analyze('app')
#analyze('gh')
#analyze('jira')
#analyze_gh()
#analyze('so')