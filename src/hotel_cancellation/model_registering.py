import mlflow
from mlflow.tracking import MlflowClient

from hotel_cancellation.configs.config import Config
from hotel_cancellation.utils import get_logger

logger = get_logger()


def get_registered_model_metric(model_name: str = Config.REGISTERED_MODEL_NAME) -> float:
    """
    Get the latest model metric from the Databricks model registry
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    highest_version = max(v.version for v in versions)
    logger.info(f"Highest version of model {model_name} is {highest_version}")

    model_version_details = client.get_model_version(name=model_name, version=str(highest_version))

    run = mlflow.get_run(model_version_details.run_id)
    metric = run.data.metrics[Config.MAIN_MODEL_METRIC]
    logger.info(f"Metric of the latest model {model_name} is {metric}")
    return metric


def get_latest_run() -> str:
    """
    Get the latest run from the Databricks model registry
    """
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(Config.EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1
    )
    latest_run = runs[0]
    return latest_run


def get_latest_run_metric() -> float:
    latest_run = get_latest_run()
    metric = latest_run.data.metrics[Config.MAIN_MODEL_METRIC]
    logger.info(f"Metric of the latest run is {metric}")
    return metric, latest_run


def register_model(latest_run: str):
    registered_model = mlflow.register_model(
        model_uri=f"runs:/{latest_run.info.run_id}/model-logistic-regression", name=Config.REGISTERED_MODEL_NAME
    )

    # set alias to the model named latest-model
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=Config.REGISTERED_MODEL_NAME, version=registered_model.version, alias="latest-model"
    )
    logger.info(f"Model registered as: {Config.REGISTERED_MODEL_NAME}")
