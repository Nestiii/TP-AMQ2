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

    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["ucimlrepo>=0.0", "awswrangler>=3.0"],
        system_site_packages=False
    )
    def obtain_original_data():
        """Descargar dataset desde la fuente original y guardarlo en MinIO (S3)."""
        import awswrangler as wr
        from ucimlrepo import fetch_ucirepo 
        
        # fetch dataset 
        airfoil_self_noise = fetch_ucirepo(id=291) 

        data_path = "s3://data/raw/airfoil_self_noise.csv"
        dataframe = airfoil_self_noise.data.original

        wr.s3.to_csv(df=dataframe, path=data_path, index=False)

    @task.virtualenv(
        task_id="clean_and_transform_data",
        requirements=[
            "awswrangler>=3.0",
        ],
        system_site_packages=False
    )
    def clean_and_transform_data():
        """Limpieza de datos y feature engineering (dummies, transformaciones)."""
        import pandas as pd
        import awswrangler as wr

        data_original_path = "s3://data/raw/airfoil_self_noise.csv"
        dataset = wr.s3.read_csv(data_original_path)

        dataframe = pd.DataFrame(dataset)
        # Solo tenemos que renombrar los nombres por default de las columnas
        dataframe = dataframe.rename(columns={
            'frequency': 'f',
            'attack-angle': 'alpha',
            'chord-length': 'c',
            'free-stream-velocity': 'U_infinity',
            'suction-side-displacemente-thickness': 'delta',
            'scaled-sound-pressure': 'SSPL'
        })

        data_clean_path = "s3://data/clean/airfoil_self_noise.csv"
        wr.s3.to_csv(df=dataframe, path=data_clean_path, index=False)

    @task.virtualenv(
        task_id="split_dataset",
        requirements=[
            "awswrangler>=3.0",
            "scikit-learn>=1.0",
        ],
        system_site_packages=True
    )
    def split_dataset():
        """Division train/test estratificada."""
        import awswrangler as wr
        from airflow.models import Variable
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        data_clean_path = "s3://data/clean/airfoil_self_noise.csv"
        dataset = wr.s3.read_csv(data_clean_path)

        dataframe = pd.DataFrame(dataset)
        target_col = Variable.get("target_col_airfoil")
        X = dataframe.drop(columns=[target_col])
        y = dataframe[[target_col]]

        test_size = Variable.get("test_size_airfoil")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        wr.s3.to_csv(df=X_train, path="s3://data/final/train/X_train.csv", index=False)
        wr.s3.to_csv(df=X_test,  path="s3://data/final/test/X_test.csv", index=False)
        wr.s3.to_csv(df=y_train, path="s3://data/final/train/y_train.csv", index=False)
        wr.s3.to_csv(df=y_test,  path="s3://data/final/test/y_test.csv", index=False)

    @task.virtualenv(
        task_id="normalize_features",
        requirements=[
            "awswrangler>=3.0",
            "scikit-learn>=1.0",
            "mlflow>=2.0",
        ],
        system_site_packages=False
    )
    def normalize_features():
        import awswrangler as wr
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        X_train = wr.s3.read_csv("s3://data/final/train/X_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/X_test.csv")

        scaler = StandardScaler()
        
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        wr.s3.to_csv(df=X_train_scaled, path="s3://data/final/train/X_train_scaled.csv", index=False)
        wr.s3.to_csv(df=X_test_scaled,  path="s3://data/final/test/X_test_scaled.csv", index=False)

    # Task dependencies: sequential pipeline
    obtain_original_data() >> clean_and_transform_data() >> split_dataset() >> normalize_features()
