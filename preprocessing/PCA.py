
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
# build covariance matrxi (d X d)
# decompose into eigenvectors and eigenvalues 
# find the 'k' principal components
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

X_train = X

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_train)

import numpy as np

X_cov = np.cov(X_train.T)
eig_val, eig_vec = np.linalg.eig(X_cov)

# Check for varince explained by the principal components
tot_eig = sum(eig_val)
var_exp = [i/tot_eig for i in eig_val] #cal variance explained ratio
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt
plt.plot(range(cum_var_exp.shape[0]),cum_var_exp, marker = 'o')
plt.grid()
plt.show()

# now we need to build the tranformation matrix which is made of eigen vectors in decreasing order of corresponding
# eigen values
#top 6 principal components expain 90% of variance, we can dump the rest

eigen_pairs = [(eig_val[i],eig_vec[:,i]) for i in range(len(eig_val))]
eigen_pairs.sort(reverse=True)

transfor_mat = np.hstack((eigen_pairs[0][1][:,np.newaxis],  #stacking the top 6 eigen vectors
                          eigen_pairs[1][1][:,np.newaxis],
                          eigen_pairs[2][1][:,np.newaxis],
                          eigen_pairs[3][1][:,np.newaxis],
                          eigen_pairs[4][1][:,np.newaxis],
                          eigen_pairs[5][1][:,np.newaxis]))
#transfor_mat

X_train_pca = X_train.dot(transfor_mat)
#the transformed data is X_train_pca with extracted features that explain around 90 % of variance
