from fastapi import FastAPI

app = FastAPI(
    title="ML Models and something more Inc.",
    description="REST API para servir predicciones de modelos ML",
    version="0.1.0",
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is running"}


@app.post("/predict/")
def predict():
    # TODO: Implementar prediccion cargando modelo champion desde MLflow
    return {"message": "Prediction endpoint - pending model integration"}
