# Trabajo Final - Aprendizaje de Máquina II (CEIA - FIUBA)

**Integrantes**: Juan Ignacio Teich | Facundo Rivas | Lourdes Florencia González Branchi | Alberto Stryfe Adriano Levano

Este proyecto implementa un entorno productivo simulado de MLOps para la empresa ficticia **"ML Models and something more Inc."**, que ofrece modelos de Machine Learning a través de una REST API. El ciclo completo de vida de un modelo ML está containerizado en Docker: desde la obtención y limpieza de datos, pasando por el entrenamiento y registro de modelos, hasta el serving de predicciones en producción y el re-entrenamiento automático.

---

## Tabla de contenidos

1. [Objetivo del trabajo práctico](#objetivo-del-trabajo-práctico)
2. [Dataset y modelo](#dataset-y-modelo)
3. [Requisitos previos](#requisitos-previos)
4. [Tecnología utilizada](#tecnología-utilizada)
5. [Arquitectura](#arquitectura).
6. [Estructura del proyecto](#estructura-del-proyecto)
7. [Guía de uso paso a paso](#guía-de-uso-paso-a-paso)
   - [Paso 1: Levantar la infraestructura](#paso-1-levantar-la-infraestructura)
   - [Paso 2: Ejecutar el ETL](#paso-2-ejecutar-el-etl)
   - [Paso 3: Entrenamiento inicial](#paso-3-entrenamiento-inicial)
   - [Paso 4: Predicción via API](#paso-4-predicción-via-api)
   - [Paso 5: Reentrenamiento automático](#paso-5-reentrenamiento-automático)
8. [Comandos útiles](#comandos-útiles)
9. [Troubleshooting](#troubleshooting)

---

## Objetivo del trabajo práctico

El objetivo es implementar el ciclo completo de MLOps en un ambiente containerizado, simulando un entorno productivo real. Esto incluye:

- **DataOps**: Pipeline ETL para obtener, limpiar y preparar datos
- **MLOps**: Entrenamiento, tracking de experimentos, registro y versionado de modelos
- **Serving**: API REST para servir predicciones del modelo en producción
- **Re-entrenamiento**: Pipeline automatizado para reentrenar y promover modelos

**Nivel elegido**: Contenedores (nota 8-10)

---

## Dataset y modelo

**Dataset**: [Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) (UCI ML Repository, id=291)

El problema consiste en predecir el **nivel de presión sonora escalado (SSPL, en dB)** que produce un perfil de ala bajo distintas condiciones aerodinámicas. Es un problema de **regresión**: dado un conjunto de características del ala y del flujo de aire, el modelo predice cuánto ruido va a generar.

### Features de entrada

| Feature | Descripción |
|---------|-------------|
| `f` | Frecuencia (Hz) |
| `alpha` | Ángulo de ataque (grados) |
| `c` | Longitud de cuerda (m) |
| `U_infinity` | Velocidad de flujo libre (m/s) |
| `delta` | Espesor de desplazamiento del lado de succión (m) |
| **`SSPL`** | **Target**: Nivel de presión sonora escalado (dB) |

---

## Requisitos previos

- [Docker](https://docs.docker.com/get-docker/) (>= 20.10) con [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y **abierto**
- [Docker Compose](https://docs.docker.com/compose/install/) (>= 2.0)
- Python >= 3.12 (solo para correr el notebook de entrenamiento desde tu máquina)
- Mínimo 4 GB de RAM asignados a Docker (Docker Desktop → Settings → Resources)
- Mínimo 2 CPUs
- Mínimo 10 GB de espacio en disco

> **Mac users**: para correr el notebook de entrenamiento necesitan `libomp` (lo pide XGBoost). Instalar con: `brew install libomp`

---

## Tecnología utilizada

| Servicio | Tecnología | Función |
|----------|------------|---------|
| Orquestador | Apache Airflow | Ejecutar y programar pipelines (ETL, reentrenamiento) |
| Tracking ML | MLflow | Tracking de experimentos, model registry, versionado |
| Data Lake | MinIO (S3) | Almacenamiento de datasets y artefactos del modelo |
| Base de datos | PostgreSQL | Metadata de Airflow y MLflow |
| API REST | FastAPI | Servir predicciones del modelo en producción |
| Broker | Valkey (Redis) | Message broker interno para Airflow CeleryExecutor |
| Containers | Docker Compose | Orquestación de todos los servicios |

---

## Estructura del proyecto

```
TP-AMQ2/
├── .env                              # Variables de entorno
├── docker-compose.yaml               # Orquestación de servicios
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
└── notebook_example/                 # Notebooks de experimentación
    ├── experiment_mlflow.ipynb       # Optuna + MLflow + registro champion
    ├── mlflow_aux.py                 # Helper get_or_create_experiment
    ├── optuna_aux.py                 # Espacio de búsqueda (RF, XGBoost, SVR)
    └── plots.py                      # Correlación + information gain
```

---

## Guía de uso paso a paso

> Los pasos deben ejecutarse **en orden**: cada uno depende del anterior. El ETL produce datos en MinIO → el notebook los lee y registra el champion en MLflow → la API carga el champion → el re-train compara contra el champion.

### Paso 1: Levantar la infraestructura

**En la terminal**, desde la carpeta raíz del proyecto, corré:

```bash
# Clonar el repositorio (si todavía no lo hiciste)
git clone <url-del-repositorio>
cd TP-AMQ2

# Levantar todos los servicios
docker compose --profile all up --build
```

La primera vez tarda varios minutos mientras se construyen las imágenes. La terminal va a quedar mostrando logs en tiempo real. Esto es normal, **no la cierres**.

#### ¿Cómo sé que todo está listo?

Abrí una **nueva terminal** y corré:

```bash
docker compose --profile all ps
```

Lo que deberías ver es que la mayoría de los servicios muestran estado `healthy`. Es normal que FastAPI aparezca como `Restarting` en este punto, se va a estabilizar una vez que exista un modelo champion (Paso 3).

También podés verificar abriendo estas URLs en el navegador:

| Servicio | URL | Credenciales |
|----------|-----|-------------|
| Airflow | http://localhost:8080 | User: airflow / Password: airflow |
| MLflow | http://localhost:5001 | — |
| MinIO | http://localhost:9001 | User: minio / Password: minio123 |
| FastAPI Docs — (disponible después del Paso 3)| http://localhost:8800/docs | — |

#### ¿Qué hay en cada servicio en este punto?

- **Airflow**: Vas a ver dos DAGs listos para correr:
  - `process_etl_data`: descarga, limpia y guarda los datos en MinIO
  - `retrain_the_model`: re-entrena el modelo y lo compara contra el champion actual
- **MinIO**: Vas a ver dos buckets creados automáticamente:
  - `data/`: donde se guardan los datos del dataset (lo llena el ETL en el Paso 2)
  - `mlflow/`: donde MLflow guarda los artefactos del modelo champion (lo llena el notebook en el Paso 3)
- **MLflow**: Vacío por ahora, se llena en los pasos siguientes

---

### Paso 2: Ejecutar el ETL

El ETL (Extract, Transform, Load) es el proceso que descarga el dataset, lo limpia y lo divide en conjuntos de entrenamiento y testeo. Los datos crudos y procesados se guardan en el bucket `data/` de MinIO.

**En Airflow** (http://localhost:8080):

1. Activá el DAG `process_etl_data` clickeando el toggle de la derecha (se pone azul)
2. Ejecutalo manualmente clickeando el botón ▷ (triángulo) a la derecha del toggle
3. Confirmá haciendo click en **Trigger DAG**
4. Esperá ~3-5 minutos hasta que el DAG muestre un tilde verde ✅

#### ¿Cómo verifico que el ETL funcionó?

Entrá a **MinIO** (http://localhost:9001) → bucket `data/` y deberías ver estas carpetas:

| Carpeta | Contenido |
|---------|-----------|
| `raw/` | Datos originales sin procesar descargados del dataset |
| `clean/` | Datos con limpieza básica aplicada |
| `final/train/` | Conjunto de entrenamiento listo para el modelo |
| `final/test/` | Conjunto de test para evaluar el modelo |
| `data_info/data.json` | Metadata del dataset |

> En este paso solo nos importa el bucket `data/`. El bucket `mlflow/` se llena en el Paso 3.

---

### Paso 3: Entrenamiento inicial

El notebook entrena el modelo por primera vez. Prueba 50 combinaciones de hiperparámetros sobre tres tipos de modelos (RandomForest, XGBoost y SVR), registra cada prueba en MLflow, y al final guarda el mejor como **"champion"** en el Model Registry.

> Este paso se corre **desde tu máquina** (NO dentro de Docker), en una **nueva terminal**, dentro de la carpeta del repositorio.

#### Crear el entorno virtual e instalar dependencias

**En Windows** (PowerShell):
```powershell
# Crear entorno virtual
python -m venv .venv

# Activar el entorno virtual
.venv\Scripts\activate

# Actualizar pip e instalar dependencias
pip install --upgrade pip
pip install jupyter awswrangler mlflow optuna scikit-learn xgboost seaborn matplotlib pandas
```

**En Mac/Linux**:
```bash
# Crear entorno virtual
python3.12 -m venv .venv

# Activar el entorno virtual
source .venv/bin/activate

# Actualizar pip e instalar dependencias
pip install --upgrade pip
pip install jupyter awswrangler mlflow optuna scikit-learn xgboost seaborn matplotlib pandas
```

#### Lanzar el notebook

```bash
jupyter notebook notebook_example/experiment_mlflow.ipynb
```

Esto abre el navegador automáticamente con el archivo `experiment_mlflow.ipynb`. Una vez abierto, ejecutá todas las celdas desde el menú:
- **Jupyter clásico**: `Cell → Run All`
- **JupyterLab**: `Run → Run All Cells`

La corrida tarda varios minutos. Lo que hace internamente:

1. Lee los datos de entrenamiento y test desde MinIO (bucket `data/`)
2. Prueba 50 combinaciones de hiperparámetros usando Optuna (cross-validation de 5-fold, métrica R²)
3. Cada prueba queda registrada como un "trial" en MLflow. Podés verlos en http://localhost:5001 → Model training → Experiments → "Airfoil Self-Noise" → Runs
4. Toma el mejor modelo, lo reentrena con todos los datos de entrenamiento, lo evalúa en el test set y lo registra en MLflow como `airfoil_model_prod` v1.
5. MLflow guarda el archivo del modelo (`model.pkl`), y otros artefactos, automáticamente en el bucket `mlflow/` de MinIO

#### Verificar que el champion quedó registrado

Entrá a **MLflow** (http://localhost:5001) → **Models** → `airfoil_model_prod`. Deberías ver la versión 1.

Una vez registrado el champion, **FastAPI se va a estabilizar automáticamente** y http://localhost:8800/docs va a estar disponible.

---

### Paso 4: Predicción via API

FastAPI expone dos endpoints:

- `GET /` → health check: confirma que la API está viva y muestra qué versión del modelo tiene cargada. No es para usuarios finales.
- `POST /predict/` → recibe las 5 features aerodinámicas y devuelve la predicción de SSPL en dB.

#### Usar la interfaz gráfica (recomendado)

Entrá a **http://localhost:8800/docs** — esta es la interfaz Swagger UI donde podés probar la API sin escribir ningún comando.

1. Hacé click en **POST /predict/**
3. Hacé click en **Try it out**
4. En el campo **Request body** ya viene un ejemplo precargado. Podés usarlo tal cual o modificar los valores
5. Hacé click en **Execute**
6. El resultado aparece abajo en **Response body**:

```json
{"sspl_db": 129.48, "model_version": 1}
```

- `sspl_db`: el nivel de presión sonora predicho en dB
- `model_version`: la versión del modelo champion que hizo la predicción

#### Usar la terminal (opcional)

**En Windows** (PowerShell):
```powershell
Invoke-RestMethod -Uri "http://localhost:8800/predict/" -Method POST -ContentType "application/json" -Body '{"features": {"f": 1000, "alpha": 5.4, "c": 0.1524, "U_infinity": 39.6, "delta": 0.00529}}'
```

**En Mac/Linux**:
```bash
curl -X POST http://localhost:8800/predict/ \
  -H "Content-Type: application/json" \
  -d '{"features": {"f": 1000, "alpha": 5.4, "c": 0.1524, "U_infinity": 39.6, "delta": 0.00529}}'
```

---

### Paso 5: Reentrenamiento automático

El DAG `retrain_the_model` re-entrena el modelo usando los datos actuales y lo compara contra el champion. Si el nuevo modelo (challenger) supera al champion en R², pasa a ser el nuevo champion y la API lo carga automáticamente sin reiniciar.

**En Airflow** (http://localhost:8080):

1. Activá el DAG `retrain_the_model` con el toggle
2. Ejecutalo con el botón ▷
3. El DAG tiene dos tareas internas:
   - `train_the_challenger_model`: entrena un nuevo modelo con los mismos hiperparámetros del champion y lo registra con el alias "challenger"
   - `evaluate_champion_challenger`: compara el R² de ambos en el test set y promueve al ganador

> **Nota**: como el challenger usa los mismos datos, hiperparámetros y semilla aleatoria que el champion, el R² suele ser casi idéntico y generalmente gana el champion (por la condición `>` estricta). Para forzar que gane el challenger, podés modificar `retrain_the_model.py` quitando el `random_state` o cambiando los hiperparámetros.

---
## Arquitectura
El siguiente diagrama muestra los servicios y flujos del sistema:

![Diagrama de arquitectura](docs/images/diagrama.png)
---

## Comandos útiles

```bash
# Levantar todo (con logs en pantalla)
docker compose --profile all up --build

# Levantar en background (sin logs)
docker compose --profile all up -d

# Ver logs de un servicio específico
docker compose logs -f fastapi

# Verificar estado de los servicios
docker compose --profile all ps

# Apagar todo (mantiene los datos)
docker compose --profile all down

# Destruir todo (borra volúmenes y datos — volvé al Paso 1)
docker compose down --volumes
```

---

## Troubleshooting

### Docker Desktop no responde / "daemon is running" error
Docker Desktop no está abierto. Abrilo desde el menú inicio y esperá a que el ícono de la ballena esté quieto antes de correr cualquier comando.

### "WSL needs updating" en Docker Desktop (Windows)
Abrí PowerShell como administrador y corré:
```powershell
wsl --update
```
Luego hacé click en "Try Again" en Docker Desktop.

### FastAPI aparece como `Restarting` después del Paso 1
Es normal. FastAPI necesita un modelo champion para arrancar. Se estabiliza automáticamente después de completar el Paso 3.

### El notebook tira `404 / NoSuchKey` al leer datos de S3
El DAG `process_etl_data` no corrió o se resetearon los volúmenes. Volvé al Paso 2.

### La API devuelve `model_version: 0` o no carga
No hay champion registrado en MLflow. Completá el Paso 3 hasta el final.

### El DAG `retrain_the_model` falla con `RestException: model not found`
El Paso 3 no se completó. El retrain requiere que ya exista un champion.

### Puerto ya en uso (`bind: address already in use`)
Algún servicio local está usando los puertos 8080, 5001, 9000, 9001, 5432 o 8800. Cerralo o cambiá el mapeo en `docker-compose.yaml`.

### Quiero empezar de cero
```bash
docker compose down --volumes
docker compose --profile all up --build
# Volvé a correr desde el Paso 2
```
