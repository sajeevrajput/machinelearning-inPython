# Demonstrating the usage of Statified KFold cross validation  for model evaluation where class proportions are preserved
# when split is done

import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                header = None)
X, y = df.iloc[:,2:].values, df.iloc[:,1].values


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

lr_pipe = Pipeline([('scl', StandardScaler()),
                   ('pcomp', PCA(n_components = 2)),
                   ('lr', LogisticRegression())])
lr_pipe.fit(X_train, y_train)

print('Logistic Regression score using Pipeline: %.3f' %lr_pipe.score(X_test, y_test))


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 10,
                      random_state = 1,) # The folds are made by preserving the percentage of samples for each class
                      
scores = []
for train_idx, test_idx in skf.split(X_train, y_train):
    lr_pipe.fit(X_train[train_idx],y_train[train_idx])
    score = lr_pipe.score(X_train[test_idx],y_train[test_idx])
    scores.append(score)
    
import numpy as np
cv_avg_score = np.mean(scores)
print(cv_avg_score)
