import numpy as np
from sklearn import datasets

iris=datasets.load_iris()
X, y = iris.data[:, [2,3]], iris.target



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)

X_train=sc.transform(X_train)
X_test=sc.transform(X_test)



from sklearn.svm import SVC
svm=SVC(kernel = 'linear', C = 1.0, random_state = 0)   #Try using different kernels like 'rbf', ‘poly’, ‘sigmoid’, ‘precomputed’
svm.fit(X_train, y_train)



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


#Call Plot Decision boundary on the entire normalized data

X_comb=np.vstack((X_train,X_test))
y_comb=np.hstack((y_train,y_test))
plot_decisionboundary(X_comb,y_comb,svm)
