import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
import joblib
import os
import json
from mlflow import MlflowClient
import mlflow.pyfunc 
from mlflow.exceptions import MlflowException

# change what model you want to use (registered model name)
MODEL_NAME = "IRIS-Classifier-LogReg"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8100")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def fetch_and_load_latest_model(model_name: str):
    """
    Fetches the latest model version from the MLflow Model Registry and loads it.
    """
    model_uri = f"models:/{model_name}/latest"
    print(f"Attempting to load latest model from URI: {model_uri}")
    
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded the latest model for '{model_name}'.")

        return loaded_model

    except MlflowException as e:
        print(f"Error loading model from registry: {e}")
        print("Ensure the model name is correct, the MLflow Tracking Server is running, and the model is registered.")
        return None
      
def fetch_and_load_best_model(model_name: str, metric_key: str = "accuracy"):
    """
    Fetches the model version with the highest value for a specific metric.
    """
    # mlflow.set_tracking_uri("http://127.0.0.1:8100")
    client = MlflowClient()
    
    all_versions = client.search_model_versions(filter_string=f"name='{model_name}'")
    if not all_versions:
        raise ValueError(f"No model versions found for registered model: {model_name}")
        
    best_acc = -1.0
    best_model_version = None
        
    for mv in all_versions:
        try:
            run = client.get_run(mv.run_id)
            current_acc = run.data.metrics.get(metric_key, -2.0)
            
            if current_acc > best_acc:
                best_acc = current_acc
                best_model_version = mv.version
                
        except MlflowException as e:
            print(f"Warning: Could not fetch run metrics for version {mv.version}: {e}")
            continue

    if best_model_version is None or best_acc == -1.0:
        raise MlflowException(f"Failed to find a suitable model version with metric '{metric_key}'.")

    print(f"Best model found: Version {best_model_version} with {metric_key}={best_acc:.4f}")

    model_uri = f"models:/{model_name}/{best_model_version}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model

def run_evaluation(model_type:str):
    """
    The main evaluation pipeline function.
    """
    if model_type=="latest":
        model = fetch_and_load_latest_model(MODEL_NAME)
    elif model_type=="best":
        model = fetch_and_load_best_model(MODEL_NAME)
    else:
        print("model type can only be 'best' or 'latest'")

    if model is None:
        print("Evaluation pipeline aborted due to failure to load model.")
        return
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc