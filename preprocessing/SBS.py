import numpy as np
from sklearn.model_selection import train_test_split
from itertools import combinations

class SBS():
    def __init__(self,estimator,k_features, test_size = 0.3):
        #intialize paramteres
        self.test_size = test_size
        self.estimator =  estimator
        
    def fit(self, X, y):
        #split train-test data
        X_train, y_train, X_test, y_test =  train_test_split(X, y,
                                                             test_size = self.test_size)
        dim = X.shape[1]
        self.indices = tuple(range(dim-1))
        score = calc_score(X_train, y_train, X_test, y_test, self.indices)
        self.scores = [score]
        self.subsets = [self.indices]
        
        while dim > k_features:
            
            for p in combinations(self.indices, dim-1):
                scores = []
                subsets = []

                score_ = train_test_split(X_train, y_train, X_test, y_test, p)
                scores.append(score_)
                subsets.append(p)
            best = np.argmax(scores)
            self.scores.append(best)
            self.indices = subsets[best]
            self.subsets.append(self.indices)
            
