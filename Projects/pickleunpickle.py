
#TO BE RUN IN SESSION WHEN OUTOFLEARNING.PY IS RUNNING

#make directories and pickle stopwords and fitted classifier

import pickle
import os

dest = os.path.join("movieclassifier","pkl_objects")
if not (os.path.exists(dest)):
    os.makedirs(dest)
pickle.dump(obj=stop,file=open(os.path.join(dest,"stopwords.pkl"),'wb'),protocol=4)
pickle.dump(obj=clf,file=open(os.path.join(dest,"classifier.pkl"),'wb'),protocol=4)





#TO BE SAVED IN VECTORIZER.PY. This would be imported later

from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir,"pkl_objects",
                                     "stopwords.pkl"),"rb"))

def tokenizer(text):
    text = re.sub("<[^>]*>","",text)   #removing tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +  ''.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2*21, 
                         preprocessor=None, 
                         tokenizer=tokenizer)



#UNPICKLE AND TEST WHETHER THE CLASSIFIER WORKS PROPERLY
import pickle
import re
import os
from vectorizer import vect
clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))


import numpy as np
label = {0:'negative', 1:'positive'}
example = ['I love this movie']
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' %(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))
