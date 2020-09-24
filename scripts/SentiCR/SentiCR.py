# Modified from and Credit to  https://github.com/senticr/SentiCR/blob/master/SentiCR/SentiCR.py
# Modified by happygirlzt

from __future__ import print_function
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import  precision_score
from sklearn.metrics import  f1_score

import time
import random
import csv
import re

import nltk
from xlrd import open_workbook
from statistics import mean

import pandas as pd
import numpy as np
import argparse

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from nltk.stem.snowball import SnowballStemmer
from imblearn.over_sampling import SMOTE

from pathlib import Path
data_folder=Path('/sa4se/data')

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

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

stemmer=SnowballStemmer("english")

def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens)
    return stems

mystop_words=[
'i', 'me', 'my', 'myself', 'we', 'our',  'ourselves', 'you', 'your',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'themselves',
 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
'and',  'if', 'or', 'as', 'until',  'of', 'at', 'by',  'between', 'into',
'through', 'during', 'to', 'from', 'in', 'out', 'on', 'off', 'then', 'once', 'here',
 'there',  'all', 'any', 'both', 'each', 'few', 'more',
 'other', 'some', 'such',  'than', 'too', 'very', 's', 't', 'can', 'will',  'don', 'should', 'now'
# keywords
 'while', 'case', 'switch','def', 'abstract','byte','continue','native','private','synchronized',
 'if', 'do', 'include', 'each', 'than', 'finally', 'class', 'double', 'float', 'int','else','instanceof',
 'long', 'super', 'import', 'short', 'default', 'catch', 'try', 'new', 'final', 'extends', 'implements',
 'public', 'protected', 'static', 'this', 'return', 'char', 'const', 'break', 'boolean', 'bool', 'package',
 'byte', 'assert', 'raise', 'global', 'with', 'or', 'yield', 'in', 'out', 'except', 'and', 'enum', 'signed',
 'void', 'virtual', 'union', 'goto', 'var', 'function', 'require', 'print', 'echo', 'foreach', 'elseif', 'namespace',
 'delegate', 'event', 'override', 'struct', 'readonly', 'explicit', 'interface', 'get', 'set','elif','for',
 'throw','throws','lambda','endfor','endforeach','endif','endwhile','clone'
]

#logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s %(levelname)s %(message)s')


emodict=[]
contractions_dict=[]


# Read in the words with sentiment from the dictionary
with open('Contractions.txt',"r") as contractions,\
     open('EmoticonLookupTable.txt',"r") as emotable:
    contractions_reader=csv.reader(contractions, delimiter='\t')
    emoticon_reader=csv.reader(emotable,delimiter='\t')

    #Hash words from dictionary with their values
    contractions_dict = {rows[0]:rows[1] for rows in contractions_reader}
    emodict={rows[0]:rows[1] for rows in emoticon_reader}

    contractions.close()
    emotable.close()

grammar= r"""
NegP: {<VERB>?<ADV>+<VERB|ADJ>?<PRT|ADV><VERB>}
{<VERB>?<ADV>+<VERB|ADJ>*<ADP|DET>?<ADJ>?<NOUN>?<ADV>?}

"""
chunk_parser = nltk.RegexpParser(grammar)


contractions_regex = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
     def replace(match):
         return contractions_dict[match.group(0)]
     return contractions_regex.sub(replace, str(s).lower())


url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def remove_url(s):
    return url_regex.sub(" ",s)

negation_words =['not', 'never', 'none', 'nobody', 'nowhere', 'neither', 'barely', 'hardly',
                     'nothing', 'rarely', 'seldom', 'despite' ]

emoticon_words=['PositiveSentiment','NegativeSentiment']

def negated(input_words):
    """
    Determine if input contains negation words
    """
    neg_words = []
    neg_words.extend(negation_words)
    for word in neg_words:
        if word in input_words:
            return True
    return False

def prepend_not(word):
    if word in emoticon_words:
        return word
    elif word in negation_words:
        return word
    return "NOT_"+word

def handle_negation(comments):
    sentences = nltk.sent_tokenize(comments)
    modified_st=[]
    for st in sentences:
        allwords = nltk.word_tokenize(st)
        modified_words=[]
        if negated(allwords):
            part_of_speech = nltk.tag.pos_tag(allwords,tagset='universal')
            chunked = chunk_parser.parse(part_of_speech)
            #print("---------------------------")
            #print(st)
            for n in chunked:
                if isinstance(n, nltk.tree.Tree):
                    words = [pair[0] for pair in n.leaves()]
                    #print(words)

                    if n.label() == 'NegP' and negated(words):
                        for i, (word, pos) in enumerate(n.leaves()):
                            if (pos=="ADV" or pos=="ADJ" or pos=="VERB") and (word!="not"):
                                modified_words.append(prepend_not(word))
                            else:
                                modified_words.append(word)
                    else:
                         modified_words.extend(words)
                else:
                    modified_words.append(n[0])
            newst =' '.join(modified_words)
            #print(newst)
            modified_st.append(newst)
        else:
            modified_st.append(st)
    return ". ".join(modified_st)



def preprocess_text(text):
    text=str(text)
    comments = text.encode('ascii', 'ignore')
    comments = expand_contractions(comments)
    comments = remove_url(comments)
    comments = replace_all(comments, emodict)
    comments = handle_negation(comments)

    return  comments


class SentimentData:
    def __init__(self, text,rating):
        self.text = text
        self.rating =rating


class SentiCR:
    # Change the training oracle
    def __init__(self, algo="GBT", training_data=None):
        self.algo = algo
        if(training_data is None):
            #self.training_data=self.read_data_from_oracle_api()
            #self.training_data=self.read_data_from_oracle_jira()
            #self.training_data=self.read_data_from_oracle_app()
            #self.training_data=self.read_data_from_oracle_cr()
            self.training_data=self.read_data_from_oracle_so()
            #self.training_data=self.read_data_from_oracle_gh()
        else:
            self.training_data = training_data
        begin=time.time()
        self.model = self.create_model_from_training_data()
        end=time.time()
        print('Training used {:.2f} seconds'.format(end-begin))
        
        
    def read_data_from_oracle_api(self):
        oracle_data=[]
        cur_train=pd.read_pickle(api_train)
        cur_train['label']=cur_train['label'].replace(-1,2)
        
        print("API Reading data from oracle..")
        for index, row in cur_train.iterrows():
            oracle_data.append(SentimentData(row['sentence'],row['label']))
        return oracle_data
    
    def read_data_from_oracle_cr(self):
        oracle_data=[]
        cur_train=pd.read_pickle(cr_train)
        cur_train['label']=cur_train['label'].replace(-1,1)
        
        print("Code Reviews Reading data from oracle..")
        
        for index, row in cur_train.iterrows():
            oracle_data.append(SentimentData(row['sentence'], row['label']))
        return oracle_data
    
    def read_data_from_oracle_app(self):    
        oracle_data=[]
        cur_train=pd.read_pickle(app_train)

        cur_train['label']=cur_train['label'].replace(-1, 2)
        print("APP Reading data from oracle..")
        
        for index, row in cur_train.iterrows():
            oracle_data.append(SentimentData(row['sentence'],row['label']))
        return oracle_data
    
    def read_data_from_oracle_gh(self):    
        oracle_data=[]
        cur_train=pd.read_pickle(gh_train)
        print("Github Reading data from oracle..")
        
        for index, row in cur_train.iterrows():
            oracle_data.append(SentimentData(row['sentence'],row['label']))
        return oracle_data
    
    def read_data_from_oracle_so(self):    
        oracle_data=[]
        cur_train=pd.read_pickle(so_train)
        cur_train['label']=cur_train['label'].replace(-1, 2)
        print("StackOverflow Reading data from oracle..")
        
        for index, row in cur_train.iterrows():
            oracle_data.append(SentimentData(row['sentence'],row['label']))
        return oracle_data
    
    def read_data_from_oracle_jira(self):    
        oracle_data=[]
        cur_train=pd.read_pickle(jira_train)
        cur_train['label']=cur_train['label'].replace(-1, 0)
        
        print("Jira Reading data from oracle..")
        
        for index, row in cur_train.iterrows():
            oracle_data.append(SentimentData(row['sentence'],row['label']))
        return oracle_data
    
    def get_classifier(self):
        algo=self.algo

        if algo=="GBT":
            return GradientBoostingClassifier()
        elif algo=="RF":
            return  RandomForestClassifier()
        elif algo=="ADB":
            return AdaBoostClassifier()
        elif algo =="DT":
            return  DecisionTreeClassifier()
        elif algo=="NB":
            return  BernoulliNB()
        elif algo=="SGD":
            return  SGDClassifier()
        elif algo=="SVC":
            return LinearSVC()
        elif algo=="MLPC":
            return MLPClassifier(activation='logistic',  batch_size='auto',
            early_stopping=True, hidden_layer_sizes=(100,), learning_rate='adaptive',
            learning_rate_init=0.1, max_iter=5000, random_state=1,
            solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
            warm_start=False)
        return 0

    def create_model_from_training_data(self):
        training_comments=[]
        training_ratings=[]
        print("Training classifier model..")
        for sentidata in self.training_data:
            comments = preprocess_text(sentidata.text)
            training_comments.append(comments)
            training_ratings.append(sentidata.rating)

        # discard stopwords, apply stemming, and discard words present in less than 3 comments
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, sublinear_tf=True, max_df=0.5,
                                     stop_words=mystop_words, min_df=3)
        X_train = self.vectorizer.fit_transform(training_comments).toarray()
        Y_train = np.array(training_ratings)
        
        #Apply SMOTE to improve ratio of the minority class
        #smote_model = SMOTE(ratio=0.5, random_state=None, k=None, k_neighbors=15, m=None, m_neighbors=15, out_step=.0001, kind='regular', svm_estimator=None, n_jobs=1)
        
        smote_model=SMOTE(random_state=2)
        X_resampled, Y_resampled=smote_model.fit_sample(X_train, Y_train)

        model=self.get_classifier()
        model.fit(X_resampled, Y_resampled)

        return model


    def get_sentiment_polarity(self,text):
        comment=preprocess_text(text)
        feature_vector=self.vectorizer.transform([comment]).toarray()
        sentiment_class=self.model.predict(feature_vector)
        return sentiment_class

    def get_sentiment_polarity_collection(self,texts):
        predictions=[]
        for text in texts:
            comment=preprocess_text(text)
            feature_vector=self.vectorizer.transform([comment]).toarray()
            sentiment_class=self.model.predict(feature_vector)
            predictions.append(sentiment_class)

        return predictions


def ten_fold_cross_validation(dataset,ALGO):
    kf = KFold(n_splits=10)

    run_precision = []
    run_recall = []
    run_f1score = []
    run_accuracy = []

    count=1

    #Randomly divide the dataset into 10 partitions
    # During each iteration one partition is used for test and remaining 9 are used for training
    for train, test in kf.split(dataset):
        print("Using split-"+str(count)+" as test data..")
        classifier_model=SentiCR(algo=ALGO,training_data= dataset[train])

        test_comments=[comments.text for comments in dataset[test]]
        test_ratings=[comments.rating for comments in dataset[test]]

        pred = classifier_model.get_sentiment_polarity_collection(test_comments)

        precision = precision_score(test_ratings, pred, pos_label=-1)
        recall = recall_score(test_ratings, pred, pos_label=-1)
        f1score = f1_score(test_ratings, pred, pos_label=-1)
        accuracy = accuracy_score(test_ratings, pred)

        run_accuracy.append(accuracy)
        run_f1score.append(f1score)
        run_precision.append(precision)
        run_recall.append(recall)
        count+=1

    return (mean(run_precision),mean(run_recall),mean(run_f1score),mean(run_accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised sentiment classifier')

    parser.add_argument('--algo', type=str,
                        help='Classification algorithm', default="GBT")


    parser.add_argument('--repeat', type=int,
                        help='Iteration count', default=100)

    args = parser.parse_args()
    ALGO = args.algo
    REPEAT = args.repeat

    print("Cross validation")
    print("Algrithm: " + ALGO)
    print("Repeat: " + str(REPEAT))

    workbook = open_workbook("oracle.xlsx")
    sheet = workbook.sheet_by_index(0)
    oracle_data = []

    for cell_num in range(0, sheet.nrows):
        comments = SentimentData(sheet.cell(cell_num, 0).value,sheet.cell(cell_num, 1).value)
        oracle_data.append(comments)

    random.shuffle(oracle_data)

    oracle_data=np.array(oracle_data)

    Precision = []
    Recall = []
    Fmean = []
    Accuracy = []

    for k in range (0,REPEAT):
        print(".............................")
        print("Run# {}".format(k))
        (precision, recall, f1score, accuracy)=ten_fold_cross_validation(oracle_data,ALGO)
        Precision.append(precision)
        Recall.append(recall)
        Fmean.append(f1score)
        Accuracy.append(accuracy)
        print("Precision:"+str(precision))
        print("Recall:" + str(recall))
        print("F-measure:" + str(f1score))
        print("Accuracy:" + str(accuracy))

    ##########################
    training = open("cross-validation-" + ALGO + ".csv", 'w')
    training.write("Run,Algo,Precision,Recall,Fscore,Accuracy\n")

    for k in range(0, REPEAT):
        training.write(str(k) + "," + ALGO + "," + str(Precision[k]) + "," + str(Recall[k]) + "," +
                       str(Fmean[k]) + "," + str(Accuracy[k]) + "\n")
    training.close()

    print("-------------------------")
    print("Average Precision: {}".format(mean(Precision)))
    print("Average Recall: {}".format(mean(Recall)))
    print("Average Fmean: {}".format(mean(Fmean)))
    print("Average Accuracy: {}".format(mean(Accuracy)))
    print("-------------------------")