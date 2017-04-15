# Demonstration of hierarchical clustering with agglomerative approach and using complete linkage as distance metric

# Generate random data set
import numpy as np
np.random.seed(123)
X = np.random.random_sample([5,3])*10

#Define labels and variable names
variables = ['V1','V2','V2']
labels = ['ID0', 'ID1', 'ID2', 'ID3', 'ID4']

import pandas as pd
df = pd.DataFrame(X, columns = variables, index = labels)

from scipy.spatial.distance import pdist
row_dist = pdist(df, metric = 'euclidean') 
#we can now use the condensed distance matrix - row_dist for agglometric clustering

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
linkage_mat = linkage(row_dist, method = 'complete') #Returns linkage matrix. 
                                                #Can use df.values also instead of pairwise distance matrix
fig=plt.figure(figsize=[8,8])
axd=fig.add_axes([0.09,1,0.2,0.6])
row_dend=dendrogram(linkage_mat, labels=labels,orientation='left')

clustered_X = X[row_dend['leaves'][::-1]]

axm=fig.add_axes([0.22,1,0.6,0.6])
cax=axm.matshow(clustered_X,cmap="hot_r")
axd.set_xticks([])
axd.set_yticks([])

axm.set_xticklabels(['']+variables)
axm.set_yticklabels(['']+row_dend['ivl'][::-1])
fig.colorbar(cax)

plt.show()
