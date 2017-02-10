import numpy as np

class Perceptron(object):
    def __init__(self,l_rate=0.01,n_iter=10):
        self.l_rate=l_rate
        self.n_iter=n_iter
    def fit(self,X,y):
        self.theta=np.zeros(X.shape[1]+1)
        self.errors_=[]
        for _ in range(self.n_iter):
            n_err=0
            for xi,yi in zip(X,y):
                update=self.l_rate*(yi-self.predict(xi))
                self.theta[0]+=update
                self.theta[1:]+=update*xi
                n_err+=np.where(update==0,0,1)
            self.errors_.append(n_err)
            print(self.errors_)
        return self
    def predict(self,xi):
        return np.where((np.dot(xi,self.theta[1:])+self.theta[0])>=0,1,-1)
