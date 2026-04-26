"""
DAG Retrain Model
Pipeline de reentrenamiento: entrena un challenger con los hiperparametros del champion
actual, lo evalua sobre el test set y promueve al ganador en el MLflow Model Registry.
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

    @task.virtualenv(
        task_id="train_the_challenger_model",
        requirements=[
            "scikit-learn>=1.3",
            "mlflow>=3.1",
            "awswrangler>=3.0",
            "xgboost>=2.0",
        ],
        system_site_packages=True,
    )
    def train_the_challenger_model():
        """
        Entrena un modelo challenger usando los hiperparametros del champion actual,
        loguea el experimento en MLflow y lo registra como `airfoil_model_prod`
        con alias `challenger`.
        """
        import datetime as dt
        import mlflow
        import awswrangler as wr

        from sklearn.base import clone
        from sklearn.metrics import r2_score
        from mlflow.models import infer_signature

        MODEL_NAME = "airfoil_model_prod"
        EXPERIMENT_NAME = "Airfoil Self-Noise"

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()

        # 1. Cargar el champion actual y clonar sus hiperparametros (sin estado entrenado)
        champion_data = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_model = mlflow.sklearn.load_model(champion_data.source)
        challenger_model = clone(champion_model)

        # 2. Cargar datos
        X_train = wr.s3.read_csv("s3://data/final/train/X_train.csv")
        y_train = wr.s3.read_csv("s3://data/final/train/y_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/X_test.csv")
        y_test = wr.s3.read_csv("s3://data/final/test/y_test.csv")

        # 3. Entrenar challenger
        challenger_model.fit(X_train, y_train.to_numpy().ravel())

        # 4. Evaluar en test
        y_pred = challenger_model.predict(X_test)
        r2 = r2_score(y_test.to_numpy().ravel(), y_pred)

        # 5. Loguear el experimento en MLflow
        experiment = mlflow.set_experiment(EXPERIMENT_NAME)
        run_name = "Challenger_run_" + dt.datetime.today().strftime("%Y/%m/%d-%H:%M:%S")

        with mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment.experiment_id,
            tags={"experiment": "challenger models", "dataset": "Airfoil Self-Noise"},
            log_system_metrics=True,
        ):
            params = {k: str(v) for k, v in challenger_model.get_params().items()}
            params["model"] = type(challenger_model).__name__
            mlflow.log_params(params)
            mlflow.log_metric("test_r2", r2)

            signature = infer_signature(X_train, challenger_model.predict(X_train))

            model_info = mlflow.sklearn.log_model(
                sk_model=challenger_model,
                name="model",
                signature=signature,
                serialization_format="cloudpickle",
                registered_model_name="airfoil_model_dev",
                metadata={"model_data_version": 1},
            )

        # 6. Registrar version como challenger en airfoil_model_prod
        tags = {k: str(v) for k, v in challenger_model.get_params().items()}
        tags["model"] = type(challenger_model).__name__
        tags["test_r2"] = f"{r2:.4f}"

        result = client.create_model_version(
            name=MODEL_NAME,
            source=model_info.model_uri,
            run_id=model_info.run_id,
            tags=tags,
        )
        client.set_registered_model_alias(MODEL_NAME, "challenger", result.version)

        print(f"Challenger registrado: v{result.version}, test_r2={r2:.4f}")

    @task.virtualenv(
        task_id="evaluate_champion_challenge",
        requirements=[
            "scikit-learn>=1.3",
            "mlflow>=3.1",
            "awswrangler>=3.0",
            "xgboost>=2.0",
        ],
        system_site_packages=True,
    )
    def evaluate_champion_challenge():
        """
        Compara R² del champion vs challenger sobre el test set. Si el challenger
        gana, lo promueve a champion (y libera el alias challenger). Si pierde,
        solo libera el alias challenger.
        """
        import mlflow
        import awswrangler as wr

        from sklearn.metrics import r2_score

        MODEL_NAME = "airfoil_model_prod"
        EXPERIMENT_NAME = "Airfoil Self-Noise"

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()

        def load_by_alias(alias):
            mv = client.get_model_version_by_alias(MODEL_NAME, alias)
            return mlflow.sklearn.load_model(mv.source), int(mv.version)

        # 1. Cargar champion y challenger
        champion_model, champion_version = load_by_alias("champion")
        challenger_model, challenger_version = load_by_alias("challenger")

        # 2. Cargar test set
        X_test = wr.s3.read_csv("s3://data/final/test/X_test.csv")
        y_test = wr.s3.read_csv("s3://data/final/test/y_test.csv")
        y_true = y_test.to_numpy().ravel()

        # 3. Evaluar ambos
        r2_champion = r2_score(y_true, champion_model.predict(X_test))
        r2_challenger = r2_score(y_true, challenger_model.predict(X_test))

        print(
            f"Champion v{champion_version} R²={r2_champion:.4f} | "
            f"Challenger v{challenger_version} R²={r2_challenger:.4f}"
        )

        # 4. Loguear el resultado de la comparacion
        experiment = mlflow.set_experiment(EXPERIMENT_NAME)
        last_runs = mlflow.search_runs(
            [experiment.experiment_id], output_format="list"
        )
        with mlflow.start_run(run_id=last_runs[0].info.run_id):
            mlflow.log_metric("test_r2_champion", r2_champion)
            mlflow.log_metric("test_r2_challenger", r2_challenger)
            winner = "Challenger" if r2_challenger > r2_champion else "Champion"
            mlflow.log_param("Winner", winner)

        # 5. Promover o demote
        if r2_challenger > r2_champion:
            print("Challenger gana → promoviendo a champion")
            client.delete_registered_model_alias(MODEL_NAME, "champion")
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
            client.set_registered_model_alias(MODEL_NAME, "champion", challenger_version)
        else:
            print("Champion mantiene su lugar → liberando alias challenger")
            client.delete_registered_model_alias(MODEL_NAME, "challenger")

    train_the_challenger_model() >> evaluate_champion_challenge()
