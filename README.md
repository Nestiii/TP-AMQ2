# Trabajo Final - Aprendizaje de Maquina II (CEIA - FIUBA)

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
- Minimo 4 GB de RAM asignados a Docker
- Minimo 2 CPUs
- Minimo 10 GB de espacio en disco

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
│   │   ├── requirements.txt
│   │   └── files/                    # Datos de ejemplo y artefactos
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
```

---

## Flujo de uso

### Paso 1: Levantar la infraestructura

```bash
docker compose --profile all up --build
```

### Paso 2: Ejecutar el ETL

1. Ir a Airflow (http://localhost:8080)
2. Activar y ejecutar el DAG `process_etl_data`
3. El pipeline descarga, limpia, divide y normaliza los datos
4. Los datos quedan almacenados en MinIO (bucket `data`)

### Paso 3: Entrenamiento inicial

1. Ejecutar el notebook `notebook_example/experiment_mlflow.ipynb`
2. Optuna busca los mejores hiperparametros
3. Cada trial se registra en MLflow
4. El mejor modelo se registra como "champion" en el Model Registry

### Paso 4: Prediccion

```bash
curl -X POST http://localhost:8800/predict/ \
  -H "Content-Type: application/json" \
  -d '{"features": { ... }}'
```

### Paso 5: Reentrenamiento

1. Ir a Airflow (http://localhost:8080)
2. Ejecutar el DAG `retrain_the_model`
3. Se entrena un challenger, se compara con el champion
4. Si el challenger es mejor, se promueve automaticamente

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

## Integrantes

| # | Nombre |
|---|--------|
| 1 | Juan Ignacio Teich |
| 2 | Facundo Rivas |
| 3 | Lourdes Florencia Gonzalez Branchi |
| 4 | Alberto Stryfe Adriano Levano |

---

## Licencia

Proyecto academico - CEIA FIUBA
