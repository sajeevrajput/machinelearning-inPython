# Code to demonstrate the use of density based apprach for clustering. The algorithm is run on moon dataset.
# We see a comparision of other types of clustering approaches with this one

from sklearn.datasets import make_moons
X,y = make_moons(n_samples=150, 
                 noise=0.05,
                 random_state=0)
#View the dataset
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1])
plt.show()



#kMeans clustering
from sklearn.cluster import KMeans
km = KMeans(init = 'random',
            max_iter=150, 
            n_clusters=2,
            random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km==0,0], X[y_km==0,1], c='green')
plt.scatter(X[y_km==1,0], X[y_km==1,1], c='red')
plt.title("KMeans")
plt.show()



#Agglomerative Clustering with complete linkage
from sklearn.cluster.hierarchical import AgglomerativeClustering
aggcl = AgglomerativeClustering(n_clusters=2,
                                linkage='complete')
y_agcl = aggcl.fit_predict(X)

plt.scatter(X[y_agcl==0,0], X[y_agcl==0,1], c='green')
plt.scatter(X[y_agcl==1,0], X[y_agcl==1,1], c='red')
plt.title("Aggolomerative Clustering")
plt.show()



#Demonstaring clustering using density-based approach
from sklearn.cluster import DBSCAN
dbs = DBSCAN(eps=0.2,
             min_samples=5)
y_dbs = dbs.fit_predict(X)

plt.scatter(X[y_dbs==0,0], X[y_dbs==0,1], c='green')
plt.scatter(X[y_dbs==1,0], X[y_dbs==1,1], c='red')
plt.title("Density based(DBSCAN) Clustering")
plt.show()
