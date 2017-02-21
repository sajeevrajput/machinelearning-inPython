from sklearn import datasets      #LOAD DATASET IRIS 
import numpy as np
iris = datasets.load_iris()
X, y = np.array(iris.data)[:, [2,3]], np.array(iris.target)

from sklearn.cross_validation import train_test_split     
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)  # Split into Test and Train data

sc = StandardScaler()                   #Scale and normalize the data
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)



ppn = Perceptron(eta0 = 0.1, n_iter = 35, random_state = 0)   #Fit the weigt parameters
ppn.fit(X_train, y_train)

y_pred = ppn.predict(X_test)             #Calculate accuracy
sum(np.where((y_pred != y_test), 1, 0))
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)




x1 = np.linspace(X[:,0].min()-1, X[:,0].max()+1,100)   #Build meshgrid and plot 
x2 = np.linspace(X[:,1].min()-1, X[:,1].max()+1,100)
xx, yy = np.meshgrid(x1, x2)
mesh_coo = np.array([xx.ravel(), yy.ravel()]).T
z = ppn.predict(mesh_coo)
z = z.reshape(xx.shape)

import matplotlib.pyplot as plt
plt.contourf(xx, yy, z)
plt.show()
