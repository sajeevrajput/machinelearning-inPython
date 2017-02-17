import numpy as np

class AdalineSGD(object):
    def __init__(self, l_rate = 0.01, n_iter = 10):
        self.l_rate = l_rate
        self.n_iter = n_iter
    def fit(self, X, y):
        self.avgcost_ = []
        self.weights_ = np.zeros(X.shape[1]+1)
        
        for _ in range(self.n_iter):
            cost_iter=[]
            for xi,yi in zip(X, y):
                
                error = yi - self.predict(xi)
                cost_iter.append(0.5 * (error**2).sum())
                
                self.weights_[1:] += self.l_rate * error * xi
                self.weights_[0] += self.l_rate * error
            self.avgcost_.append(sum(cost_iter)/len(y))
        print(self.avgcost_)
        #print(self.weights_)
        return self
    def predict(self,xi):
        return np.where((np.dot(xi,self.weights_[1:]) + self.weights_[0] >= 0), 1, -1)        
"""
STARTER CODE
*************
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("D:/iris.csv",header=None)
df=df.iloc[:100,[0,2,4]]
X=df.iloc[:,[0,1]].values
y=df.iloc[:,2].values
y=np.where(y=='Iris-setosa',1,-1)


Xmean=X.mean(axis=0)                        # Normalized X
Xstd=X.std(axis=0)
for dim in np.arange(X.ndim):
    X[:,dim]=(X[:,dim]-Xmean[dim])/Xstd[dim]

a=AdalineSGD()
a.fit(X,y)
plt.plot(np.arange(a.n_iter),a.avgcost_)
plt.show()

"""
