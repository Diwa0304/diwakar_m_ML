FROM python:3.10-slim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/encoder.joblib models/
COPY evaluate_models.py .

EXPOSE 8000

ENV MLFLOW_TRACKING_URI="http://136.119.19.191:8100/"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
