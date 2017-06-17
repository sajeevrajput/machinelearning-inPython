import pyprind
import pandas as pd
import os

"""
#TO BUILD DATASET FROM UNZIPPED FILE

df = pd.DataFrame()
pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg':0}

for s in ('test', 'train'):
    for l in ('neg','pos'):
        path = 'E:/DATASETS/aclImdb/%s/%s' %(s,l)
        #print (path)
        for file in os.listdir(path):
            with open(os.path.join(path,file),'r', encoding='latin1') as infile:
                txt = infile.read()
                df = df.append([[txt,labels[l]]],ignore_index=True)
                pbar.update()
df.columns = ['review', 'sentiment']
"""
import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_review.csv',index=False)


#Develop preprocessor that cleans up unwanted characters
import re
def preprocessor(text):
    text = re.sub("<.+>","",text) #substitutes all tags <> characters with " "
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +  ''.join(emoticons).replace('-', '')
    return text

df['review'] = df['review'].apply(preprocessor)

# Storing Training and Test data
X_train, y_train = df.iloc[:25000,0].values, df.iloc[:25000,1].values

X_test, y_test = df.iloc[25000:,0].values, df.iloc[25000:,1].values


#define tokenizer, tokenizer_porter and stop

def tokenizer(text):
    return text.split()

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

#create parameter grid for pipeline
param_grid = [{'vect__ngram_range':(1,1),
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer,tokenizer_porter],
               'lreg_penalty': ['l1', 'l2'],
               'lreg__C': [1.0, 10.0, 100.0]},
              
              {'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer,tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'lreg__penalty': ['l1', 'l2'],
               'lreg__C': [1.0, 10.0, 100.0]}]
               
#create pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

lr_tfidf= Pipeline([('vect', TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)),
                   ('lreg', LogisticRegression(random_state=0))])
                   
from sklearn.grid_search import GridSearchCV
gs_lr_tfidf = GridSearchCV(estimator=lr_tfidf,
                  param_grid= param_grid,
                  cv=5,
                  scoring = 'accuracy',
                  n_jobs = -1)
gs_lr_tfidf.fit(X_train,y_train)

