# Demonstration of hierarchical clustering with agglomerative approach and using complete linkage as distance metric

# Generate random data set
import numpy as np
np.random.seed(123)
X = np.random.random_sample([5,3])*10

#Define labels and variable names
#variables = ['V1','V2','V2']
#labels = ['ID0', 'ID1', 'ID2', 'ID3', 'ID4']

#import pandas as pd
#df = pd.DataFrame(X, columns = variables, index = labels)

from scipy.spatial.distance import pdist
row_dist = pdist(X, metric = 'euclidean') # can use dataframe - df as well

#we can now use the condensed distance matrix - row_dist for agglometric clustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
#Returns linkage matrix. Can use df.values also instead of pairwise distance matrix
linkage_mat = linkage(row_dist, method = 'complete') 

row_dend=dendrogram(linkage_mat, labels=labels)
plt.ylabel('Euclidean Distance')
plt.show()
