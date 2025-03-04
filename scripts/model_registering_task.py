import mlflow

from hotel_cancellation.model_registering import get_latest_run_metric, get_registered_model_metric, register_model

# load latest model run of experimetn /Shared/hotel-cancellation
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

registered_model_metric = get_registered_model_metric()
latest_run_metric, latest_run = get_latest_run_metric()
if latest_run_metric >= registered_model_metric:
    print("New model is better than the registered model. Registering the new model.")
    register_model(latest_run)
else:
    print("New model is not better than the registered model. New model is not registered")
