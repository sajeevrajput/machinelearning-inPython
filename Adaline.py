import numpy as np

class Adaline(object):
    def __init__(self, l_rate = 0.01, n_iter = 50):
        self.l_rate = l_rate
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.cost = []
        error = []
        self.theta = np.zeros(X.shape[1] + 1)
        for _ in range(self.n_iter):
            diff = y-self.predict(X)
            #print (diff)
            
            self.theta[1:] += self.l_rate * np.dot(X.T, diff)
            self.theta[0] += self.l_rate * diff.sum()
            error.append(np.where(diff == 0, 0, 1).sum())
            self.cost.append(0.5 * (diff ** 2).sum())
        #print(error)
        #print (self.cost)
        #print (self.theta)
        return self
    
    def predict(self, X):
        return (np.dot(X, self.theta[1:]) + self.theta[0])

       
""" STARTER CODE 

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = None)
df = df.iloc[:100, [0,2,4]]
X = df.iloc[:, [0,1]].values
y = df.iloc[:, 2].values
y = np.where(y == 'Iris-setosa', 1, -1)

"""
