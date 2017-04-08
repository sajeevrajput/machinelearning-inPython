########### CREATE DATASET ##########

from sklearn.datasets import make_blobs

#create 3 cluster data and store the data in X. y is the target variable
X, y = make_blobs(n_samples = 150, n_features = 2, centers = 3, cluster_std = 0.5, random_state = 0)

#View the data set
import matplotlib.pyplot as plt
import numpy as np
plt.scatter(X[:,0], X[:,1], c = 'green', marker = 'o')
plt.grid()
plt.show()

#Run kMeans clustering algorithm(default is kMeans++, else use init='random' for Llyod's algoritm or kMeans)
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, n_init = 10, max_iter = 150, tol = 1e-04, random_state = 0)
y_km = km.fit_predict(X)

#plot the data sets grouped into respective clusters by colors
# the coordinates of cluster centroids is stored in object attribute cluster_centers_
plt.scatter(X[y_km==0,0], X[y_km==0,1], color='blue')
plt.scatter(X[y_km==1,0], X[y_km==1,1], color='red')
plt.scatter(X[y_km==2,0], X[y_km==2,1], color='yellow')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250, color = 'pink', marker='*')
plt.grid()
plt.show()


# One of the challenge in kMeans or kMeans++ algorithm is that we need to know the no of clusters in advance which is not 
# feasible to guess when the dimensions are more than 3 or sometimes even when lesser.
# We can figure out the optimum no of clusters by monitoring the SSE or sum of squared errors generated for each instance of
# cluster number chosen. This should give an elbow shaped structure which would let us know the optimum no of clusters.
# Fortunately, the instance from scikit kmeans has an attribute called 'inertia_' that holds this value
"""
sse_=[]
for i in range(1,11):
  km = KMeans(n_clusters = i, n_init = 10, max_iter = 150, tol = 1e-04, random_state = 0)
  km.fit(X)
  sse_.append(km.inertia_)
plt.plot(np.arange(1,11),sse_)
plt.show()
"""
