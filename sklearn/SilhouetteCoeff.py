# Build and view dataset

from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples = 150, n_features = 2, 
               centers = 4, cluster_std = 0.5, random_state = 0)


import matplotlib.pyplot as plt
import numpy as np

clr=['red','blue', 'green', 'yellow']

for i in np.unique(y):
    plt.scatter(X[y==i, 0], X[y==i, 1], c=clr[i])
plt.show()



# Run KMeans on dataset(notice init='random') and see the clusters
from sklearn.cluster import KMeans
kmplus = KMeans(init = 'random', max_iter = 150, n_init = 10, 
                n_clusters = 4, random_state = 0, tol=1e-4)
y_pred=kmplus.fit_predict(X)

for i in np.unique(y_pred):
    plt.scatter(X[y_pred==i, 0], X[y_pred==i, 1], c=clr[i])
plt.show()


#Plot sihouette coefficients for each sample clusterwise
lbound=0
yticks=[]
for i in np.unique(y_pred):
    #to find the sil coeff for each group of clusters
    i_pred_samples=len(y_pred[y_pred==i])
    i_sil_vals=sil_vals[y_pred==i]
    i_sil_vals.sort()
    
    yticks.append(lbound+i_pred_samples/2)
    
    plt.barh(bottom = range(lbound, lbound + i_pred_samples), width = i_sil_vals,
             height=1, color=clr[i],edgecolor='none')
    lbound += i_pred_samples
    
sil_avg=np.mean(sil_vals)
plt.axvline(sil_avg, color = 'magenta', linestyle = '--')
plt.yticks(yticks, range(len(yticks)))
plt.xlabel('Silhouette Coefficients')
plt.ylabel('Clusters')

plt.show()
# Silouhette Coefficient close to 1 is good-dense clustering
