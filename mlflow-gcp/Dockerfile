FROM ghcr.io/mlflow/mlflow:v3.10.0-full

RUN pip install google-cloud-storage

CMD mlflow server \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --artifacts-destination ${MLFLOW_ARTIFACTS_DESTINATION} \
    --serve-artifacts
