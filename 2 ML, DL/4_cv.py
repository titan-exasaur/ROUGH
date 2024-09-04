import numpy as np
import pandas as pd
import seaborn as sns

data = sns.load_dataset("iris")
# data['Species'] = data['Species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=10)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)

results = cross_val_score(lr, X, y, cv=kfold)
print(results)

from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=10)

results = cross_val_score(lr, X, y, cv=skfold)
print(results)