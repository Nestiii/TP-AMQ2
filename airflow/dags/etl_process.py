"""
DAG ETL Process
Pipeline de datos: obtencion, limpieza, feature engineering, split y normalizacion.
Los datos se almacenan en MinIO (S3).
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
    dag_id="process_etl_data",
    default_args=default_args,
    description="ETL pipeline: obtain, clean, split and normalize data",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["etl", "data"],
) as dag:

    @task
    def obtain_original_data():
        """Descargar dataset desde la fuente original y guardarlo en MinIO (S3)."""
        # TODO: Implementar descarga del dataset y subida a S3
        # Ejemplo:
        #   import boto3
        #   s3_client = boto3.client("s3")
        #   s3_client.upload_file(local_path, "data", "raw/dataset.csv")
        logger.info("Task: obtain_original_data - PLACEHOLDER")
        logger.info("Pending: download dataset and upload to s3://data/raw/")

    @task
    def clean_and_transform_data():
        """Limpieza de datos y feature engineering (dummies, transformaciones)."""
        # TODO: Implementar limpieza y feature engineering
        # - Eliminar duplicados
        # - Manejar nulos
        # - Variables dummy / one-hot encoding
        logger.info("Task: clean_and_transform_data - PLACEHOLDER")
        logger.info("Pending: clean data, handle nulls, create dummy variables")

    @task
    def split_dataset():
        """Division train/test estratificada."""
        # TODO: Implementar split estratificado
        # - Leer datos limpios de S3
        # - Split con sklearn train_test_split
        # - Guardar train y test en S3
        logger.info("Task: split_dataset - PLACEHOLDER")
        logger.info("Pending: stratified train/test split and save to S3")

    @task
    def normalize_features():
        """Normalizar features numericas con StandardScaler."""
        # TODO: Implementar normalizacion
        # - Leer train/test de S3
        # - Fit scaler en train, transform train y test
        # - Guardar datos normalizados y scaler en S3
        logger.info("Task: normalize_features - PLACEHOLDER")
        logger.info("Pending: normalize numeric features and save to S3")

    # Task dependencies: sequential pipeline
    obtain_original_data() >> clean_and_transform_data() >> split_dataset() >> normalize_features()
