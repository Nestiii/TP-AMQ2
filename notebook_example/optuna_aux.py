import mlflow
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score


REGRESSOR_NAMES = ["RandomForest", "XGBoost", "SVR"]


def champion_callback(study, frozen_trial):
    """
    Callback de logging que reporta cuando un nuevo trial mejora al mejor anterior.
    """
    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def build_model_from_params(params):
    """
    Reconstruye un estimador a partir del diccionario de mejores parametros de un study.
    """
    name = params["regressor"]

    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_split=params["rf_min_samples_split"],
            random_state=42,
            n_jobs=-1,
        )

    if name == "XGBoost":
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=params["xgb_n_estimators"],
            max_depth=params["xgb_max_depth"],
            learning_rate=params["xgb_learning_rate"],
            subsample=params["xgb_subsample"],
            random_state=42,
            n_jobs=-1,
        )

    if name == "SVR":
        return SVR(
            kernel="rbf",
            C=params["svr_c"],
            gamma="scale",
            epsilon=params["svr_epsilon"],
        )

    raise ValueError(f"Unknown regressor: {name}")


def objective(trial, X_train, y_train, experiment_id):
    """
    Optimiza hiperparametros sobre RandomForest, XGBoost y SVR.
    Devuelve R2 medio en CV (5 folds), maximizar.
    """
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"Trial: {trial.number}", nested=True
    ):
        params = {"objective": "reg:r2", "eval_metric": "r2"}

        regressor_name = trial.suggest_categorical("regressor", REGRESSOR_NAMES)
        params["regressor"] = regressor_name

        if regressor_name == "RandomForest":
            params["rf_n_estimators"] = trial.suggest_int("rf_n_estimators", 20, 300)
            params["rf_max_depth"] = trial.suggest_int("rf_max_depth", 3, 32, log=True)
            params["rf_min_samples_split"] = trial.suggest_int("rf_min_samples_split", 2, 20)
            model = RandomForestRegressor(
                n_estimators=params["rf_n_estimators"],
                max_depth=params["rf_max_depth"],
                min_samples_split=params["rf_min_samples_split"],
                random_state=42,
                n_jobs=-1,
            )

        elif regressor_name == "XGBoost":
            params["xgb_n_estimators"] = trial.suggest_int("xgb_n_estimators", 50, 500)
            params["xgb_max_depth"] = trial.suggest_int("xgb_max_depth", 3, 12)
            params["xgb_learning_rate"] = trial.suggest_float(
                "xgb_learning_rate", 0.01, 0.3, log=True
            )
            params["xgb_subsample"] = trial.suggest_float("xgb_subsample", 0.6, 1.0)
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=params["xgb_n_estimators"],
                max_depth=params["xgb_max_depth"],
                learning_rate=params["xgb_learning_rate"],
                subsample=params["xgb_subsample"],
                random_state=42,
                n_jobs=-1,
            )

        else:  # SVR
            params["svr_c"] = trial.suggest_float("svr_c", 0.1, 200, log=True)
            params["svr_epsilon"] = trial.suggest_float("svr_epsilon", 0.01, 1.0, log=True)
            model = SVR(
                kernel="rbf",
                C=params["svr_c"],
                gamma="scale",
                epsilon=params["svr_epsilon"],
            )

        score = cross_val_score(
            model, X_train, y_train.to_numpy().ravel(), n_jobs=-1, cv=5, scoring="r2"
        )

        mlflow.log_params(params)
        mlflow.log_metric("r2_cv", score.mean())

    return score.mean()
