import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Perceptron

"""
The idea is to create a meshgrid where horizontal and vertical axes will correspond to first and second feature
 respectively. The coordinates of meshgrid(2,no_of_grids) are then fed into Pereptron.predict() method where they are classified binarily.
 The binary output as a whole cuts the meshgrid in half- one for each outcome of classification. 
 Finally, the training data is scattered on the mesh grid to see how the decison boundary fares.

#---------Building Meshplot---------
# horizontal width and vertical width of mesh grid calculated
# from input data columns
# Mesh grid generated and then their coordinates are build(10000,2)
# and passed into predict() method. The outcome vector(10000,) is  
# reshaped(100,100) for contour plot

"""
from matplotlib.colors import ListedColormap

def plot_decisionboundary(X, y, classifier, resolution = 0.02):
    markers = ['o', '*', 's', '#', '^', '+']
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
    cmap=ListedColormap(colors[:len(np.unique(y))])
    
    #BUILD Meshplot
    x1, x2 = X[:,0], X[:,1]
    xx, yy = np.meshgrid(np.arange(x1.min()-1, x1.max()+1, resolution), np.arange(x2.min()-1, x2.max()+1, resolution))
    mesh_coo = np.array([xx.ravel(), yy.ravel()]).T
    z = classifier.predict(mesh_coo)
    zz = z.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap=cmap)
    
    #scatter plot input data
    for idx, target in enumerate(np.unique(y)):
        plt.scatter(X[y==target][:,0], X[y==target][:,1], c=cmap(idx), marker=markers[idx], label=target)
    
    plt.legend(loc = "upper left")
    plt.show()

    

#=====STARTER CODE======
#Load IRIS Dataset

from sklearn import datasets
iris=datasets.load_iris()
X,y=np.array(iris.data)[:,[2,3]],np.array(iris.target)


# In case you use an online data-set
"""
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None)
df = df.iloc[:100,[0,2,4]]                  # load only the first 2 sets of flower with their sepal length and petal length
X = df.iloc[:,[0,1]].values                 # X is input data (100,2)
y = df.iloc[:,2].values                     # y is label
y = np.where(y=='Iris-setosa', 1, -1)       # Flower Setosa classified as 1 and -1, otherwise

"""

#Normalize and Run Perceptron fit from sciki-learn

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0) #Split into Train/Test data

sc=StandardScaler()   #Run normalization
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)

ppn=Perceptron(eta0=0.1,n_iter=40,random_state=0)
ppn.fit(X_train,y_train)



#Call Plot Decision boundary on the entire normalized data
X_comb=np.vstack((X_train,X_test))
y_comb=np.hstack((y_train,y_test))
plot_decisionboundary(X_comb,y_comb,ppn)




