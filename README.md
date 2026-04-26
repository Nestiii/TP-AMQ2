# Trabajo Final - Aprendizaje de Maquina II (CEIA - FIUBA)

**Integrantes**: Juan Ignacio Teich | Facundo Rivas | Lourdes Florencia Gonzalez Branchi | Alberto Stryfe Adriano Levano

Entorno productivo simulado de MLOps para la empresa ficticia **"ML Models and something more Inc."**, que ofrece modelos de Machine Learning a traves de una REST API.

El proyecto implementa el ciclo completo de vida de un modelo ML en un ambiente containerizado (Docker):

- **DataOps**: Pipeline ETL para obtener, limpiar y preparar datos
- **MLOps**: Entrenamiento, tracking de experimentos, registro y versionado de modelos
- **Serving**: API REST para servir predicciones del modelo en produccion
- **Re-entrenamiento**: Pipeline automatizado para reentrenar y promover modelos

**Nivel elegido**: Contenedores (nota 8-10)

---

## Arquitectura

```
┌───────────────────────────────────────────────────────────────┐
│                       Docker Compose                          │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌───────────┐│
│  │  Airflow    │  │  MLflow    │  │  MinIO   │  │ PostgreSQL ││
│  │  :8080      │  │  :5001     │  │  :9001   │  │  :5432     ││
│  │             │  │            │  │  (S3)    │  │            ││
│  │ - Scheduler │  │ - Tracking │  │          │  │ - mlflow_db││
│  │ - Webserver │  │ - Registry │  │ - /data  │  │ - airflow  ││
│  │ - Worker    │  │            │  │ - /mlflow│  │            ││
│  └────────────┘  └────────────┘  └──────────┘  └───────────┘│
│                                                               │
│  ┌───────────────────────────────────────────────────────────┐│
│  │                   FastAPI  :8800                           ││
│  │          Endpoint POST /predict/ (produccion)             ││
│  └───────────────────────────────────────────────────────────┘│
│                                                               │
│  ┌────────────┐                                               │
│  │   Valkey    │  (broker interno de Airflow)                 │
│  └────────────┘                                               │
└───────────────────────────────────────────────────────────────┘
```

### Comunicacion entre servicios

```
                    ┌─────────┐
                    │  MinIO  │
                    │  (S3)   │
                    └────┬────┘
                         │ lee/escribe datos
            ┌────────────┼────────────┐
            │            │            │
       ┌────▼────┐  ┌────▼────┐  ┌───▼─────┐
       │ Airflow │  │Notebook │  │  FastAPI │
       │  DAGs   │  │Training │  │   API    │
       └────┬────┘  └────┬────┘  └───┬─────┘
            │            │            │
            │  registra  │  registra  │ carga modelo
            │  metadata  │  modelos   │ champion
            └────────────┼────────────┘
                    ┌────▼────┐
                    │  MLflow │
                    │ Registry│
                    └────┬────┘
                         │ metadata
                    ┌────▼────┐
                    │Postgres │
                    └─────────┘
```

---

## Stack tecnologico

| Servicio | Tecnologia | Funcion |
|----------|------------|---------|
| Orquestador | Apache Airflow | Ejecutar y programar pipelines (ETL, reentrenamiento) |
| Tracking ML | MLflow | Tracking de experimentos, model registry, versionado |
| Data Lake | MinIO (S3) | Almacenamiento de datasets y artefactos |
| Base de datos | PostgreSQL | Metadata de Airflow y MLflow |
| API REST | FastAPI | Servir predicciones del modelo en produccion |
| Broker | Valkey (Redis) | Message broker para Airflow CeleryExecutor |
| Containers | Docker Compose | Orquestacion de todos los servicios |

---

## Requisitos previos

- [Docker](https://docs.docker.com/get-docker/) (>= 20.10)
- [Docker Compose](https://docs.docker.com/compose/install/) (>= 2.0)
- Python >= 3.12 (solo para correr el notebook de entrenamiento desde tu maquina)
- Minimo 4 GB de RAM asignados a Docker
- Minimo 2 CPUs
- Minimo 10 GB de espacio en disco

> **Mac users**: para correr el notebook de entrenamiento necesitan `libomp` (lo pide XGBoost). Instalar con: `brew install libomp`

---

## Instalacion y setup

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd TP-AMQ2
```

### 2. Levantar todos los servicios

```bash
docker compose --profile all up --build
```

La primera vez tarda varios minutos mientras se construyen las imagenes.

### 3. Verificar que todo esta corriendo

```bash
docker compose --profile all ps
```

Todos los servicios deben mostrar estado `healthy`.

### URLs de acceso

| Servicio | URL | Credenciales |
|----------|-----|-------------|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5001 | - |
| MinIO | http://localhost:9001 | minio / minio123 |
| FastAPI | http://localhost:8800 | - |
| FastAPI Docs | http://localhost:8800/docs | - |

---

## Estructura del proyecto

```
TP-AMQ2/
├── .env                              # Variables de entorno
├── docker-compose.yaml               # Orquestacion de servicios
├── dockerfiles/
│   ├── airflow/                      # Imagen Airflow custom
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── fastapi/                      # API REST
│   │   ├── Dockerfile
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── mlflow/                       # MLflow tracking server
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── postgres/                     # PostgreSQL con init script
│       ├── Dockerfile
│       └── mlflow.sql
├── airflow/
│   ├── dags/                         # DAGs de Airflow
│   │   ├── etl_process.py            # Pipeline ETL
│   │   └── retrain_the_model.py      # Pipeline de reentrenamiento
│   └── secrets/                      # Conexiones y variables
│       ├── connections.yaml
│       └── variables.yaml
└── notebook_example/                 # Notebooks de experimentacion
    ├── experiment_mlflow.ipynb       # Optuna + MLflow + registro champion
    ├── mlflow_aux.py                 # Helper get_or_create_experiment
    ├── optuna_aux.py                 # Espacio de busqueda (RF, XGBoost, SVR)
    └── plots.py                      # Correlacion + information gain
```

---

## Flujo de uso

### Paso 1: Levantar la infraestructura

```bash
docker compose --profile all up --build
```

> Los pasos deben ejecutarse en orden: cada uno depende del anterior (ETL produce datos en S3 → notebook los lee y registra champion → API carga el champion → retrain compara contra el champion).

### Paso 2: Ejecutar el ETL

1. Ir a Airflow (http://localhost:8080), login `airflow / airflow`
2. Activar y disparar el DAG `process_etl_data`
3. La primera corrida tarda ~3-5 min (Airflow crea virtualenvs por task)
4. Verificar en MinIO (http://localhost:9001) que el bucket `data/` tiene `raw/`, `clean/`, `final/train/`, `final/test/`, `data_info/data.json`

### Paso 3: Entrenamiento inicial (notebook)

El notebook se corre desde tu maquina (NO dentro de Docker), apuntando a los servicios en `localhost`.

```bash
# Crear venv con Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install jupyter awswrangler mlflow optuna scikit-learn xgboost seaborn matplotlib pandas

# Lanzar Jupyter
jupyter notebook notebook_example/experiment_mlflow.ipynb
```

En el notebook, menu `Cell -> Run All`. La corrida hace:

1. Carga de `X_train/X_test/y_train/y_test` desde S3
2. Optuna prueba 50 trials sobre RandomForest, XGBoost y SVR (CV de 5-fold, metrica R²)
3. Cada trial queda como child run en el experimento "Airfoil Self-Noise" en MLflow
4. El mejor modelo se reentrena con todo el train, se evalua en test y se registra como `airfoil_model_prod` v1 con alias **`champion`** en el Model Registry
5. Verificar en MLflow UI (http://localhost:5001) -> Models -> `airfoil_model_prod`

### Paso 4: Prediccion via API

La API carga el champion al startup (y hace hot-reload despues de cada peticion si el alias cambia).

```bash
# Health check (muestra version del champion cargado)
curl http://localhost:8800/

# Prediccion
curl -X POST http://localhost:8800/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "f": 1000,
      "alpha": 5.4,
      "c": 0.1524,
      "U_infinity": 39.6,
      "delta": 0.00529
    }
  }'
# -> {"sspl_db": 129.48, "model_version": 1}
```

Documentacion interactiva en http://localhost:8800/docs (Swagger UI).

### Paso 5: Reentrenamiento

1. Ir a Airflow (http://localhost:8080)
2. Activar y disparar el DAG `retrain_the_model`
3. La task `train_the_challenger_model` clona los hiperparametros del champion, entrena con los datos actuales y registra una nueva version con alias `challenger`
4. La task `evaluate_champion_challenge` compara R² de ambos en el test set y promueve al ganador (libera el alias del perdedor)
5. Si el challenger gana, la API hace hot-reload solo en la siguiente peticion (sin reiniciar el container)

> **Caveat**: el challenger usa los mismos hiperparametros + datos + seed que el champion -> R² casi identico -> en general gana el champion por la condicion `>` estricta. Para forzar la rama "challenger gana", modificar `train_the_challenger_model` para introducir variacion (e.g. quitar `random_state`, cambiar hiperparametros, usar bootstrap).

---

## Comandos utiles

```bash
# Levantar todo
docker compose --profile all up --build

# Levantar en background
docker compose --profile all up -d

# Levantar con modo debug (incluye Airflow CLI)
docker compose --profile all --profile debug up

# Ver logs de un servicio
docker compose logs -f <servicio>

# Apagar todo (mantiene datos)
docker compose --profile all down

# Destruir todo (borra volumes, imagenes, datos)
docker compose down --rmi all --volumes
```

---

## Troubleshooting

### `XGBoostError: Library libxgboost.dylib could not be loaded` (Mac)
Falta el runtime de OpenMP. Solucion:
```bash
brew install libomp
```

### El notebook tira `404 / NoSuchKey` al leer `s3://data/...`
El DAG `process_etl_data` no corrio (o se reseteo el volumen de MinIO). Volver al Paso 2.

### La API arranca pero `/` devuelve `model_version: 0` o crashea
No hay champion registrado en MLflow. Volver al Paso 3 y correr el notebook hasta el final (ultima celda registra el champion).

### El DAG `retrain_the_model` falla con `RestException: model not found`
No existe `airfoil_model_prod` en el registry. El retrain requiere que ya haya un champion (Paso 3 completado).

### Puerto ya en uso (`bind: address already in use`)
Algun servicio local esta usando 8080/5001/9000/9001/5432/8800. Apagarlo o cambiar el mapeo en `docker-compose.yaml`.

### Quiero empezar de cero
```bash
docker compose down --volumes      # borra todos los datos
docker compose --profile all up --build
# y volver a correr Paso 2 -> 3 -> 4 -> 5
```

---

## Dataset y modelo

**Dataset**: [Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) (UCI ML Repository, id=291)

Problema de **regresion**: predecir el nivel de presion sonora escalado (SSPL) a partir de caracteristicas aerodinamicas de perfiles de alas.

| Feature | Descripcion |
|---------|-------------|
| `f` | Frecuencia (Hz) |
| `alpha` | Angulo de ataque (grados) |
| `c` | Longitud de cuerda (m) |
| `U_infinity` | Velocidad de flujo libre (m/s) |
| `delta` | Espesor de desplazamiento del lado de succion (m) |
| **`SSPL`** | **Target**: Nivel de presion sonora escalado (dB) |

---

## Licencia

Proyecto academico - CEIA FIUBA
