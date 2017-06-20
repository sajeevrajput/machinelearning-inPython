#Defining complementary functions
#1. TOKENIZER
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub("<[^>]*>","",text)   #removing tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +  ''.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized
    
#2. STREAM_DOCS - generator function that feeds one document at a time

def stream_docs(path):
    with open(path) as csv:
        next(csv) #skip header
        for row in csv:
            text, label = row[:-2],int(row[-2])
            yield text,label

#3. GET_MINI_BATCH
def get_mini_batch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y
    
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21, 
                         preprocessor=None, 
                         tokenizer=tokenizer)
clf = SGDClassifier(loss = 'log',
                    random_state= 1,
                    n_iter=1)
doc_stream = stream_docs(path = './movie_review.csv')


#main caller
import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0,1])

for _ in range(45):
    X_train, y_train = get_mini_batch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train,y_train, classes = classes)
    pbar.update()


X_test, y_test = get_mini_batch(doc_stream, size = 5000)
X_test = vect.transform(X_test)
print ('Accuracy = %s' %clf.score(X_test, y_test))
