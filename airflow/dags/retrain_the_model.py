"""
DAG Retrain Model
Pipeline de reentrenamiento: entrena un modelo challenger, lo compara
con el champion actual y promueve al ganador.
"""

import logging
from datetime import datetime

from airflow.sdk import DAG, task

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="retrain_the_model",
    default_args=default_args,
    description="Retrain pipeline: train challenger, evaluate vs champion, promote winner",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["retrain", "mlflow"],
) as dag:

    @task
    def train_the_challenger_model():
        """Entrenar un modelo challenger con los datos actuales en S3."""
        # TODO: Implementar entrenamiento del challenger
        # - Cargar datos normalizados desde S3
        # - Entrenar modelo con hiperparametros del champion o nuevos
        # - Registrar experimento en MLflow
        logger.info("Task: train_the_challenger_model - PLACEHOLDER")

    @task
    def evaluate_champion_challenge():
        """Comparar F1-score del challenger vs champion y promover al ganador."""
        # TODO: Implementar evaluacion y promocion
        # - Cargar metricas del champion desde MLflow
        # - Comparar con metricas del challenger
        # - Si challenger > champion, promover challenger a champion
        logger.info("Task: evaluate_champion_challenge - PLACEHOLDER")

    train_the_challenger_model() >> evaluate_champion_challenge()
