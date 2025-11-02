from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os
import pandas as pd
import csv
from datetime import datetime
from mlflow_utils import setup_mlflow

import mlflow
from mlflow.models.signature import infer_signature

data_file_name = os.environ.get("DATA_FILE", "iris_1.csv")
DATA_PATH = os.path.join("data", data_file_name)
print(f"Loading data file: {data_file_name}")

try:
    df = pd.read_csv(DATA_PATH)
    print("read data")
except FileNotFoundError:
    print(f"ERROR: Data file not found at {DATA_PATH}.")
    print(f"Did you remember to run 'dvc pull data/{data_file_name}'?")
    exit(1)

# set up the mlflow server to dedicated ip address
print("connecting mlflow")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")
if not MLFLOW_URI:
    # Fallback for local testing outside of CI, or just raise an error
    print("ERROR: MLFLOW_TRACKING_URI environment variable is not set.")
    exit(1)
setup_mlflow(experiment_name="logistic_regression",host_address = MLFLOW_URI)

encoder = OrdinalEncoder()
df["target"] = encoder.fit_transform(df[["species"]]).astype(int) 
X = df.drop(columns=["species", "target"]) 
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "max_iter": 500,
    "penalty": "elasticnet",
    "C": 0.5,
    "solver": "saga",
    "l1_ratio": 0.5,
}

model = LogisticRegression(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_metric = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy_metric)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/iris_classifier.joblib")
joblib.dump(encoder, "models/encoder.joblib")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
'''
# MLFlow loggings
with mlflow.start_run(run_name="Logistic-regression-elasticnet-C0.5"):
    mlflow.log_params(params)
    mlflow.log_metric("accuracy",accuracy_metric)
    mlflow.set_tag("Training info","logistic regression with elasticnet regularization for iris")
    
    signature = infer_signature(X_train, model.predict(X_train))
    
    mlflow.sklearn.log_model(
        sk_model = model,
        artifact_path = "iris_classifier",
        signature = signature,
        input_example=X_train.head(5),
        registered_model_name="IRIS-Classifier-LogReg",
    )
    print("Successfully logged run to MLflow.")
    
    
    
# metrics saving to file 
METRICS_FILE = "metrics.csv"
run_timestamp = datetime.now().isoformat()

try:
    is_new_file = not os.path.exists(METRICS_FILE) or os.stat(METRICS_FILE).st_size == 0
    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header only if the file is new/empty
        if is_new_file:
            writer.writerow(["timestamp", "metric", "value"])
            
        # Write the new metric line
        writer.writerow([run_timestamp, "accuracy", accuracy_metric])

    print(f"Metrics logged to {METRICS_FILE}")

except Exception as e:
    print(f"Error logging metrics: {e}")
'''