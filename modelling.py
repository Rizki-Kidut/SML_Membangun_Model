import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Modelling_SML_Heart_Disesase")

target_col = 'num'

df_train = pd.read_csv("heart_disease_train.csv")

X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col]

df_test = pd.read_csv("heart_disease_test.csv")

X_test = df_test.drop(columns=[target_col])
y_test = df_test[target_col]

input_example = X_train[0:5]

with mlflow.start_run():
    # Log parameters
    n_estimators = 100
    max_depth = 20
        
    mlflow.autolog()
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
   
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)