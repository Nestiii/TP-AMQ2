import mlflow


def get_or_create_experiment(experiment_name):
    """
    Recupera el ID de un experimento existente en MLflow o crea uno nuevo si no existe.
    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
