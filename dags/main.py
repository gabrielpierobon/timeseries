from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.papermill.operators.papermill import PapermillOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('run_notebook',
         default_args=default_args,
         description='Run TSMixerFull-3.ipynb notebook',
         schedule_interval=timedelta(days=1),  # Set your own schedule here
         start_date=datetime(2023, 10, 7),
         template_searchpath='/home/jovyan/work',
         catchup=False) as dag:

    run_notebook_task = PapermillOperator(
        task_id='run_notebook',
        input_nb='/home/jovyan/work/TSMixerFull-3.ipynb',  # Updated path
        output_nb='/home/jovyan/work/outputs/runs/TSMixerFull-3_output_{execution_date}.ipynb',  # Updated path
        parameters={"execution_date": "{{ ds }}"}  # Pass parameters to your notebook here
    )
