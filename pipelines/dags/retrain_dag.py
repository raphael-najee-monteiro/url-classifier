"""Airflow retraining DAG."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from pipelines.tasks.fetch import fetch_urls
from pipelines.tasks.features import extract_features


def train_and_register():
    from src.training.train import main
    main()


with DAG(
    "url_classifier_retrain",
    start_date=datetime(2026, 4, 15),
    schedule_interval=timedelta(days=1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:
    t1 = PythonOperator(task_id="fetch", python_callable=fetch_urls)
    t2 = PythonOperator(task_id="extract", python_callable=extract_features)
    t3 = PythonOperator(task_id="train_register", python_callable=train_and_register)
    t1 >> t2 >> t3
