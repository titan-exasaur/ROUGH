import os
import sys
import mlflow
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("artifacts", exist_ok=True)

data = sns.load_dataset("iris")

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35

with mlflow.start_run():
    # Preprocessing
    data['species'] = data['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Model training
    elasticnet_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    elasticnet_model.fit(X_train, y_train)

    # Save and log the model
    model_path = 'artifacts/elasticnet_model.pkl'
    joblib.dump(elasticnet_model, model_path)
    mlflow.log_artifact(model_path)

    # Predictions and metrics
    y_pred = elasticnet_model.predict(X_test)

    mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, y_pred)))
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("r2", r2_score(y_test, y_pred))

