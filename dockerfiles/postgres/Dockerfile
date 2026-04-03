FROM postgres:13

# Copy init script to create mlflow_db on first startup
COPY mlflow.sql /docker-entrypoint-initdb.d/mlflow.sql
