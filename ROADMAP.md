# Roadmap - Trabajo Final AMQ2

## Etapa 1: Infraestructura y DevOps
**Estado: Completada**

- [x] Configurar `docker-compose.yaml` con todos los servicios
- [x] Crear Dockerfiles (PostgreSQL, MLflow, Airflow, FastAPI)
- [x] Configurar variables de entorno (`.env`)
- [x] Configurar secretos y conexiones de Airflow
- [x] Crear buckets en MinIO (`data`, `mlflow`)
- [x] Verificar conectividad entre servicios
- [x] Estructura de carpetas del repositorio

## Etapa 2: Pipeline ETL (DAG de Airflow)
**Estado: Completada**

- [x] Elegir dataset: Airfoil Self-Noise (UCI ML Repository, id=291)
- [x] Implementar task `obtain_original_data`: descarga del dataset y subida a S3
- [x] Implementar task `clean_and_transform_data`: limpieza, nulos, renombrado de columnas
- [x] Implementar task `split_dataset`: division train/test (80/20)
- [x] Implementar task `normalize_features`: StandardScaler en features numericas
- [x] Guardar esquema de datos y parametros del scaler en `s3://data/data_info/data.json`
- [x] Registrar metadata del procesamiento en MLflow (dataset, params del scaler)
- [x] Testear DAG completo en Airflow

**Archivos involucrados:**
- `airflow/dags/etl_process.py`
- `airflow/secrets/variables.yaml` (actualizar target_col y variables del dataset)

## Etapa 3: Entrenamiento y Experimentacion
**Estado: Completada**

- [x] Desarrollar notebook de entrenamiento con MLflow tracking
- [x] Implementar busqueda de hiperparametros con Optuna
- [x] Definir metrica de optimizacion (ej. RMSE, R2)
- [x] Registrar experimentos en MLflow (parametros, metricas, artefactos)
- [x] Registrar el mejor modelo como "champion" en Model Registry
- [x] Crear scripts auxiliares (mlflow_aux.py, optuna_aux.py, plots.py)
- [x] Generar visualizaciones de resultados

**Archivos involucrados:**
- `notebook_example/experiment_mlflow.ipynb`
- `notebook_example/mlflow_aux.py`
- `notebook_example/optuna_aux.py`
- `notebook_example/plots.py`

## Etapa 4: API de Prediccion + Reentrenamiento
**Estado: Completada**

- [x] Implementar endpoint `POST /predict/` en FastAPI
  - [x] Cargar modelo champion desde MLflow
  - [x] Validacion de entrada con Pydantic
  - [x] Manejo de errores
- [x] Implementar DAG de reentrenamiento
  - [x] Task: entrenar modelo challenger
  - [x] Task: evaluar champion vs challenger (RMSE, R2)
  - [x] Task: promover ganador en Model Registry
- [x] Testear API end-to-end

**Archivos involucrados:**
- `dockerfiles/fastapi/app.py`
- `airflow/dags/retrain_the_model.py`

## Etapa 5: Documentacion y Entrega Final
**Estado: Pendiente**

- [ ] Completar README con ejemplos de uso
- [ ] Documentar endpoints de la API con ejemplos de request/response
- [ ] Agregar docstrings y comentarios en el codigo
- [ ] Testing end-to-end del sistema completo
- [ ] Verificar que `docker compose --profile all up` levanta todo correctamente desde cero

## Entregas

| Entrega | Fecha | Alcance |
|---------|-------|---------|
| **Primera entrega (Clase 5)** | TBD | Etapas 1 + 2 completadas: infra Docker + ETL funcional + datos en MinIO |
| **Entrega final** | 7 dias despues de la ultima clase | Sistema completo end-to-end (Etapas 1-5) |
