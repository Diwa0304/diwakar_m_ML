from evaluate_models import run_evaluation, fetch_and_load_latest_model, fetch_and_load_best_model
import joblib
import pandas as pd
import numpy as np
import os
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8100")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "IRIS-Classifier-LogReg"
ENCODER_PATH = "models/encoder.joblib" 

FEATURE_NAMES = [ "sepal_length", "sepal_width", "petal_length", "petal_width" ]

app = FastAPI()

model = None
encoder = None

try:
    model = fetch_and_load_best_model(MODEL_NAME)
    encoder = joblib.load(ENCODER_PATH)

    if model is None:
        raise RuntimeError("Failed to load model from MLflow Registry. Check MLflow server status.")

    print("API Model and Encoder loaded successfully.")
except FileNotFoundError:
    print(f"Error: Encoder file not found at {ENCODER_PATH}. Ensure train.py was run and the file is in the Docker image.")
    raise
except RuntimeError as e:
    print(f"Deployment failed: {e}")
    raise
except Exception as e:
    print(f"Fatal error during startup: {e}")
    raise
    
class FeatureInput(BaseModel):
    features: List[List[float]]    

@app.get('/health')
def health_check():
    """Simple endpoint to check if the API is running and ready."""
    return {"status": "ok", "model_loaded": (model is not None)}, 200

@app.post('/predict')
async def predict(input_data: FeatureInput):
    """
    Handles POST requests with JSON data for prediction.
    """
    try:
        df = pd.DataFrame(input_data.features, columns=FEATURE_NAMES)
        prediction_int = model.predict(df)
        prediction_species = encoder.inverse_transform(prediction_int.reshape(-1, 1)).flatten().tolist()
        
        return {
            "status": "success",
            "predictions": prediction_species
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app,host='0.0.0.0', port=8000)