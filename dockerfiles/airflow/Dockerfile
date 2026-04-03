FROM apache/airflow:3.0.2

COPY requirements.txt /requirements.txt

USER root
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --upgrade pip
RUN pip install "apache-airflow[virtualenvs]==3.0.2"
RUN pip install --no-cache-dir -r /requirements.txt
