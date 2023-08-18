import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pendulum
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from utils import *


with DAG(
    dag_id="mlops_pipeline",
    default_args=DefaultConfig.DEFAULT_DAG_ARGS,
    schedule_interval="@once",
    start_date=pendulum.datetime(2022, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops_pipeline"],
) as dag:
    
    data_extraction_task = DockerOperator(
        task_id="data_extraction_task",
        command="bash -c 'cd src && python data_extraction.py'",
        **DefaultConfig.DEFAULT_DOCKER_OPERATOR_ARGS,
    )

    model_training_task = DockerOperator(
        task_id="model_training_task",
        command="bash -c 'cd src && python model_training.py'",
        **DefaultConfig.DEFAULT_DOCKER_OPERATOR_ARGS,
    )

    model_evaluation_task = DockerOperator(
        task_id="model_evaluation_task",
        command="bash -c 'cd src && python model_evaluation.py'",
        **DefaultConfig.DEFAULT_DOCKER_OPERATOR_ARGS,
    )

    (
        data_extraction_task
        >> model_training_task
        >> model_evaluation_task
    )