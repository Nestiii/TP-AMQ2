import json
import logging
import pickle

import boto3
import mlflow
import numpy as np
import pandas as pd

from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing_extensions import Annotated


MODEL_NAME = "airfoil_model_prod"
MODEL_ALIAS = "champion"
MLFLOW_TRACKING_URI = "http://mlflow:5000"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("airfoil-api")


def load_model(model_name: str, alias: str):
    """
    Carga el modelo champion desde MLflow Registry y los parametros del scaler desde S3.
    Si MLflow no responde, cae a un model.pkl local. Si el data.json no esta en S3, cae al local.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(model_name, alias)
        model_obj = mlflow.sklearn.load_model(mv.source)
        version = int(mv.version)
        logger.info(f"Loaded model from MLflow: {model_name} v{version} (alias={alias})")
    except Exception as e:
        logger.warning(f"Falling back to local model.pkl ({e})")
        with open("/app/files/model.pkl", "rb") as f:
            model_obj = pickle.load(f)
        version = 0

    try:
        s3 = boto3.client("s3")
        s3.head_object(Bucket="data", Key="data_info/data.json")
        body = s3.get_object(Bucket="data", Key="data_info/data.json")["Body"].read().decode()
        data_info = json.loads(body)
        logger.info("Loaded data.json from S3")
    except Exception as e:
        logger.warning(f"Falling back to local data.json ({e})")
        with open("/app/files/data.json", "r") as f:
            data_info = json.load(f)

    data_info["standard_scaler_mean"] = np.array(data_info["standard_scaler_mean"])
    data_info["standard_scaler_std"] = np.array(data_info["standard_scaler_std"])
    return model_obj, version, data_info


def check_model():
    """
    Verifica si la version del champion cambio en el registry y, en ese caso, recarga modelo + scaler.
    Se ejecuta en background despues de cada prediccion.
    """
    global model, version_model, data_dict
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        new_mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        new_version = int(new_mv.version)
        if new_version != version_model:
            logger.info(f"Champion changed: v{version_model} -> v{new_version}, reloading")
            model, version_model, data_dict = load_model(MODEL_NAME, MODEL_ALIAS)
    except Exception as e:
        logger.debug(f"check_model skipped ({e})")


class ModelInput(BaseModel):
    """
    Input del modelo: 5 features aerodinamicas crudas. La API aplica el StandardScaler
    de la pipeline de entrenamiento internamente. Rangos basados en el dataset NASA
    Airfoil Self-Noise (UCI id=291).
    """

    f: float = Field(
        description="Frecuencia (Hz)",
        ge=0,
        le=25000,
    )
    alpha: float = Field(
        description="Angulo de ataque (grados)",
        ge=0,
        le=30,
    )
    c: float = Field(
        description="Longitud de cuerda del perfil (m)",
        gt=0,
        le=1.0,
    )
    U_infinity: float = Field(
        description="Velocidad del flujo libre (m/s)",
        gt=0,
        le=100,
    )
    delta: float = Field(
        description="Espesor de desplazamiento de la capa limite en el borde de fuga (m)",
        ge=0,
        le=0.5,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "f": 1000,
                    "alpha": 5.4,
                    "c": 0.1524,
                    "U_infinity": 39.6,
                    "delta": 0.00529,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Output del modelo: nivel de presion sonora escalada (SSPL) en dB.
    """

    sspl_db: float = Field(description="Prediccion de SSPL en decibeles")
    model_version: int = Field(description="Version del modelo champion usada")

    model_config = {
        "json_schema_extra": {
            "examples": [{"sspl_db": 125.43, "model_version": 1}]
        }
    }


# Cargar modelo al startup
model, version_model, data_dict = load_model(MODEL_NAME, MODEL_ALIAS)

app = FastAPI(
    title="Airfoil Self-Noise Predictor",
    description="REST API para predecir SSPL en perfiles de ala usando el modelo champion del MLflow Registry",
    version="1.0.0",
)


@app.get("/")
async def read_root():
    return JSONResponse(
        content=jsonable_encoder(
            {
                "message": "Airfoil Self-Noise Predictor API",
                "model": MODEL_NAME,
                "model_version": version_model,
            }
        )
    )


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[ModelInput, Body(embed=True)],
    background_tasks: BackgroundTasks,
):
    """
    Predice SSPL (dB) a partir de las 5 features aerodinamicas.
    Aplica el StandardScaler de la pipeline de entrenamiento antes de inferir.
    En background, chequea si el champion del registry cambio y recarga si es necesario.
    """
    feature_names = data_dict["columns"]
    raw = np.array([[getattr(features, c) for c in feature_names]], dtype=float)

    scaled = (raw - data_dict["standard_scaler_mean"]) / data_dict["standard_scaler_std"]
    features_df = pd.DataFrame(scaled, columns=feature_names)

    prediction = float(model.predict(features_df)[0])

    background_tasks.add_task(check_model)

    return ModelOutput(sspl_db=prediction, model_version=version_model)
