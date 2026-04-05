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
        system_site_packages=True
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
            "mlflow>=2.0",
        ],
        system_site_packages=True
    )
    def clean_and_transform_data():
        """Limpieza de datos y feature engineering (dummies, transformaciones)."""
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow

        import pandas as pd
        import awswrangler as wr

        from airflow.models import Variable

        data_original_path = "s3://data/raw/airfoil_self_noise.csv"
        dataset = wr.s3.read_csv(data_original_path)

        dataframe = pd.DataFrame(dataset)

        # Limpiar duplicados y nulos
        dataframe.drop_duplicates(inplace=True, ignore_index=True)
        dataframe.dropna(inplace=True, ignore_index=True)

        # Renombrar columnas
        dataframe = dataframe.rename(columns={
            'frequency': 'f',
            'attack-angle': 'alpha',
            'chord-length': 'c',
            'free-stream-velocity': 'U_infinity',
            'suction-side-displacement-thickness': 'delta',
            'scaled-sound-pressure': 'SSPL'
        })

        data_clean_path = "s3://data/clean/airfoil_self_noise.csv"
        wr.s3.to_csv(df=dataframe, path=data_clean_path, index=False)

        # Guardar metadata del dataset en S3
        client = boto3.client('s3')

        data_dict = {}
        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                raise e

        target_col = Variable.get("target_col_airfoil")
        dataset_features = dataframe.drop(columns=target_col)

        data_dict['columns'] = dataset_features.columns.to_list()
        data_dict['target_col'] = target_col
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_features.dtypes.to_dict().items()}
        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')

        data_string = json.dumps(data_dict, indent=2)
        client.put_object(Bucket='data', Key='data_info/data.json', Body=data_string)

        # Registrar en MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Airfoil Self-Noise")

        mlflow.start_run(
            run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S'),
            experiment_id=experiment.experiment_id,
            tags={"experiment": "etl", "dataset": "Airfoil Self-Noise"},
            log_system_metrics=True
        )

        mlflow_dataset = mlflow.data.from_pandas(
            dataframe,
            source="https://archive.ics.uci.edu/dataset/291/airfoil+self+noise",
            targets=target_col,
            name="airfoil_self_noise_clean"
        )
        mlflow.log_input(mlflow_dataset, context="Dataset")

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

        test_size = float(Variable.get("test_size_airfoil"))

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
        system_site_packages=True
    )
    def normalize_features():
        """Normalizar features numericas con StandardScaler."""
        import json
        import mlflow
        import boto3
        import botocore.exceptions

        import awswrangler as wr
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        X_train = wr.s3.read_csv("s3://data/final/train/X_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/X_test.csv")

        sc_X = StandardScaler(with_mean=True, with_std=True)
        X_train_arr = sc_X.fit_transform(X_train)
        X_test_arr = sc_X.transform(X_test)

        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_arr, columns=X_test.columns)

        wr.s3.to_csv(df=X_train, path="s3://data/final/train/X_train.csv", index=False)
        wr.s3.to_csv(df=X_test, path="s3://data/final/test/X_test.csv", index=False)

        # Persistir parametros del scaler en S3
        client = boto3.client('s3')

        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            raise e

        data_dict['standard_scaler_mean'] = sc_X.mean_.tolist()
        data_dict['standard_scaler_std'] = sc_X.scale_.tolist()
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(Bucket='data', Key='data_info/data.json', Body=data_string)

        # Logear en MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Airfoil Self-Noise")

        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param("Standard Scaler feature names", sc_X.feature_names_in_)
            mlflow.log_param("Standard Scaler mean values", sc_X.mean_)
            mlflow.log_param("Standard Scaler scale values", sc_X.scale_)

    # Task dependencies: sequential pipeline
    obtain_original_data() >> clean_and_transform_data() >> split_dataset() >> normalize_features()
