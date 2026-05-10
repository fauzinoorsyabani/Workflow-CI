import sys
# Trick untuk Python 3.12 agar mengenali distutils dari setuptools
try:
    import setuptools
except ImportError:
    pass

import pandas as pd
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Inisialisasi DagsHub
# dagshub.init(repo_owner='fauzinoorsyabani', repo_name='Membangun_Model_Fauzi-Noor-Syabani', mlflow=True)

def hyperparameter_tuning():
    # 1. Load Data
    df = pd.read_csv('../data/data.csv')
    X = df.drop('fail', axis=1)
    y = df['fail']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 2. Setup Grid Search
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    
    # 3. MLflow Tracking for Tuning
    mlflow.set_experiment("Machine_Failure_Tuning")
    
    with mlflow.start_run(run_name="GridSearch_RF"):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
        grid_search.fit(X_train_scaled, y_train)
        
        # Log parameter terbaik
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best Parameters: {best_params}")
        print(f"Best Accuracy: {best_score:.4f}")
        
        # Log to MLflow
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        mlflow.log_metric("best_accuracy", best_score)
        
        # Simpan Model Terbaik
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
        
        # Tambahkan artefak tambahan (Manual Logging - Syarat Advanced)
        # Kita simpan hasil tuning ke CSV sebagai artefak
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv("tuning_results.csv", index=False)
        mlflow.log_artifact("tuning_results.csv")

if __name__ == "__main__":
    hyperparameter_tuning()

