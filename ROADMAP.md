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
**Estado: Pendiente**

- [ ] Elegir dataset (de Aprendizaje de Maquina I)
- [ ] Implementar task `obtain_original_data`: descarga del dataset y subida a S3
- [ ] Implementar task `clean_and_transform_data`: limpieza, nulos, feature engineering
- [ ] Implementar task `split_dataset`: division train/test estratificada
- [ ] Implementar task `normalize_features`: StandardScaler en features numericas
- [ ] Guardar esquema de datos (JSON) en S3
- [ ] Registrar metadata del procesamiento en MLflow
- [ ] Testear DAG completo en Airflow

**Archivos involucrados:**
- `airflow/dags/etl_process.py`
- `airflow/secrets/variables.yaml` (actualizar target_col y variables del dataset)

## Etapa 3: Entrenamiento y Experimentacion
**Estado: Pendiente**

- [ ] Desarrollar notebook de entrenamiento con MLflow tracking
- [ ] Implementar busqueda de hiperparametros con Optuna
- [ ] Definir metrica de optimizacion (ej. F1-score)
- [ ] Registrar experimentos en MLflow (parametros, metricas, artefactos)
- [ ] Registrar el mejor modelo como "champion" en Model Registry
- [ ] Crear scripts auxiliares (mlflow_aux.py, optuna_aux.py, plots.py)
- [ ] Generar visualizaciones de resultados

**Archivos involucrados:**
- `notebook_example/experiment_mlflow.ipynb`
- `notebook_example/mlflow_aux.py`
- `notebook_example/optuna_aux.py`
- `notebook_example/plots.py`

## Etapa 4: API de Prediccion + Reentrenamiento
**Estado: Pendiente**

- [ ] Implementar endpoint `POST /predict/` en FastAPI
  - [ ] Cargar modelo champion desde MLflow
  - [ ] Validacion de entrada con Pydantic
  - [ ] Manejo de errores
- [ ] Implementar DAG de reentrenamiento
  - [ ] Task: entrenar modelo challenger
  - [ ] Task: evaluar champion vs challenger (F1-score)
  - [ ] Task: promover ganador en Model Registry
- [ ] Testear API end-to-end

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
