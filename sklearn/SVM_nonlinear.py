#To demonstrate use of SVM for the case of non-linearly separable data
import numpy as np


#Build Non-linearly separable Data

np.random.seed(0)
X=np.random.randn(200,2)
y = np.logical_xor(X[:, 0]>0, X[:, 1]>0)
y = np.where(y == True, 1, -1)


#Plot Dataset

import matplotlib.pyplot as plt
plt.scatter(X[y == 1, 0], X[y == 1,  1], color = 'b', label = 1)
plt.scatter(X[y == -1, 0], X[y == -1, 1], color = 'r', label = -1)
plt.legend()
plt.show()


#Run RBF-kernel based SVM classifier

from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', C = 10.0, gamma=10.0,random_state = 0)  #Vary the values of Gamma and Kernels to see the effects
svm.fit(X, y)


#### Plot decision boundary ######

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

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


#Call Plot Decision boundary

plot_decisionboundary(X,y,svm)
