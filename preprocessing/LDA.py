#Linear Discriminant Analysis

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

# scale and standardize
# compute mean vector for each class [hence supervised]
# compute scatter matrices for both between-classes and within class
# compute the eigen values and corresponding eigen vector for the vector SW_inv.dot(SB)
# find the top eigen vector (max is usually no of classes -1 ) and use this as transformation matrix

#SCALE AND STANDARDIZATION

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_train)

import numpy as np

mean_vec = []
for i in np.unique(y_train):
    mean_vec.append(np.mean(X_train[np.array(y_train==i)],axis=0))
    
#Creating scatter matrices
# 1. Within class
d = X.shape[1]    # no of features

#1. Within class scatter matrix
S_W = np.zeros((d,d))
for j in np.unique(y_train):
    
    #S_W += mean_vec[j].reshape(d,1).dot(mean_vec[j].reshape(d,1).T)    #a normalized version of it is covariance matrix
    S_W += np.cov(X_train[np.array(y_train==j)].T)

#2. Between class scatter matrix
S_B = np.zeros((d,d))
mean_overall = np.mean(X_train,axis=0)
mean_overall = mean_overall.reshape(d,1)

for index,k in enumerate(np.unique(y_train)):
    n = X_train[np.array(y_train == k)].shape[0]
    S_B = n * (mean_vec[index].reshape(d,1) - mean_overall).dot((mean_vec[index].reshape(d,1) - mean_overall).T )


#Build transformation matrix, from eigen vectors of SW.inv*SB

eig_val, eig_vec = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
