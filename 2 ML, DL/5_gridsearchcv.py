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

param_grid={
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

from sklearn.svm import SVC
svm = SVC(probability=True)

from sklearn.model_selection import GridSearchCV
GS = GridSearchCV(param_grid=param_grid,
                  estimator=svm,
                  verbose=4)
GS.fit(X_train, y_train)

print(GS.best_estimator_)
print(GS.best_params_)
print(GS.best_score_)