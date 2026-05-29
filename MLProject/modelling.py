import sys
# Trick untuk Python 3.12 agar mengenali distutils dari setuptools
try:
    import setuptools
except ImportError:
    pass

import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_baseline():
    """
    Basic modelling menggunakan MLflow autolog.
    Model baseline RandomForestClassifier tanpa hyperparameter tuning.
    """
    # 1. Load Data
    df = pd.read_csv('data/data.csv')
    X = df.drop('fail', axis=1)
    y = df['fail']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. MLflow Tracking - Autolog
    # Avoid set_experiment so it doesn't conflict with mlflow run
    mlflow.sklearn.autolog()

    # mlflow run already starts a run, so we can just use active run or start one safely by catching it
    active_run = mlflow.active_run()
    if not active_run:
        mlflow.start_run(run_name="Baseline_RF")
    
    # Model baseline tanpa hyperparameter tuning
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluasi
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Baseline Accuracy: {accuracy:.4f}")
    
    if not active_run:
        mlflow.end_run()


if __name__ == "__main__":
    train_baseline()
