import numpy as np
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.base import clone
from sklearn.metrics import accuracy_score


class SBS():
    def __init__(self,estimator,k_features, test_size = 0.3, scoring = accuracy_score):
        #intialize paramteres
        self.test_size = test_size
        self.estimator =  clone(estimator)
        self.k_features = k_features
        self.scoring = scoring
        
    def fit(self, X, y):
        #split train-test data
        X_train, X_test, y_train, y_test =  train_test_split(X, y,
                                                             test_size = self.test_size)
        dim = X.shape[1]
        self.indices = tuple(range(dim-1))
        score = self.calc_score(X_train, y_train, X_test, y_test, self.indices)
        self.scores = [score]
        self.subsets = [self.indices]
        
        while dim > self.k_features:
            
            for p in combinations(self.indices, dim-1):
                scores = []
                subsets = []

                score_ = self.calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score_)
                subsets.append(p)
            best = np.argmax(scores)
            self.scores.append(scores[best])
            self.indices = subsets[best]
            self.subsets.append(self.indices)
            
            dim-=1
            return self
        
    def calc_score(self, X_train, y_train, X_test, y_test, p):
        self.estimator.fit(X_train[:, p], y_train)
        y_pred = self.estimator.predict(X_test[:, p])
        
        score = self.scoring(y_test, y_pred)
        return score

# LOAD DATASET
import pandas as pd
df = pd.read_csv("C:/wine.csv", header = None)
df.columns = ['Class label', 'Alcohol', 
            'Malic acid', 'Ash', 
            'Alcalinity of ash', 'Magnesium', 
            'Total phenols', 'Flavanoids',
            'Nonflavanoid phenols', 
            'Proanthocyanins', 
            'Color intensity', 'Hue', 
            'OD280/OD315 of diluted wines', 
            'Proline']
X,y = df.iloc[:,1:], df.iloc[:,0]

#BUILD VALIDATION SET
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#RUN SBS Algo for feature selection
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)

sbs = SBS(estimator=knn, k_features=1)
sbs.fit(X_test_std,y_test)
