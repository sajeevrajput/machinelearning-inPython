import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Perceptron

"""
The idea is to create a meshgrid where horizontal and vertical axes will correspond to first and second feature
 respectively. The coordinates of meshgrid(2,no_of_grids) are then fed into Pereptron.predict() method where they are classified binarily.
 The binary output as a whole cuts the meshgrid in half- one for each outcome of classification. 
 Finally, the training data is scattered on the mesh grid to see how the decison boundary fares.
"""

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None)
df = df.iloc[:100,[0,2,4]]                  # load only the first 2 sets of flower with their sepal length and petal length
X = df.iloc[:,[0,1]].values                 # X is input data (100,2)
y = df.iloc[:,2].values                     # y is label
y = np.where(y=='Iris-setosa', 1, -1)       # Flower Setosa classified as 1 and -1, otherwise

p = Perceptron()          #create a Perceptron object
p.fit(X, y)               # fitting the object to get the weight vector-theta attribute that will be used when p.predict() is called

x1 = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)                   # horizontal width and vertical width of mesh grid calculated
x2 = np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100)                   # from input data columns
XX1, XX2 = np.meshgrid(x1, x2)                                          # Mesh grid generated and then their coordinates are build(10000,2)
mesh_coo = np.array([XX1.ravel(), XX2.ravel()]).T                       # and passed into predict() method. The outcome vector(10000,) is  
mesh_pred = p.predict(mesh_coo)                                         # reshaped(100,100) for contour plot
mesh_pred = mesh_pred.reshape(XX1.shape)
plt.contourf(XX1, XX2, mesh_pred)

plt.scatter(X[:50,0], X[:50,1], color="blue", label='setosa', marker="*")        # the training data is scattered on the mesh grid
plt.scatter(X[50:,0], X[50:,1], color="red", label='versicolor', marker= "o")
plt.xlabel("Sepal length (in cm)")
plt.ylabel("Petal length (in cm)")
plt.legend(loc="upper left")

plt.show()
