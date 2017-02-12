import numpy as np

class Perceptron(object):
    """
    Creates and return a Perceptron object
    __init__(): takes in learning rate and no of iterations(with a default value of 0.01 and 10 respectively) to perform
                    assigns them to instance variables
    fit(X,y):   takes in training data set X of shape (no_of_samples,no_of_features) and and corresponding outcome vector
                    y of shape (no_of_samples,)
                instance variable theta is the weight vector that would be fit after the iterations
                returns a Perceptron object
    predict(xi):takes in a training sample and predicts its outcome
    
    """
    
    def __init__(self,l_rate=0.01,n_iter=10):
        self.l_rate=l_rate
        self.n_iter=n_iter
    
    
    def fit(self,X,y):
        """
        Used to train input data
        self.errors_ : list ot hold the no of errors in each epoch
        self.theta:    single dimensional array to hold weights
        """
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
            #print(self.errors_)
        return self
    
    
    def predict(self,xi):
        """
        Predicts/Classifies output as 1 or -1 depending upon whether dot X.y >=0
        """
        return np.where((np.dot(xi,self.theta[1:])+self.theta[0])>=0,1,-1)
