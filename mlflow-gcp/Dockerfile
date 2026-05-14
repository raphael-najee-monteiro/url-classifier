FROM ghcr.io/mlflow/mlflow:v3.10.0-full

RUN pip install google-cloud-storage psycopg2-binary

CMD exec mlflow server \
    --host 0.0.0.0 \
    --port $PORT \
    --backend-store-uri $DATABASE_URL \
    --artifacts-destination $MLFLOW_ARTIFACTS_DESTINATION \
    --serve-artifacts \
    --allowed-hosts "*" \
    --no-auth