from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def run_training_pipeline():
    from train import main as train_main
    train_main()


with DAG(
    dag_id="diabetes_risk_training",
    description="Train and track the diabetes risk prediction model",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["diabetes", "machine-learning"],
) as dag:

    train_task = PythonOperator(
        task_id="train_and_register_pipeline",
        python_callable=run_training_pipeline,
    )
