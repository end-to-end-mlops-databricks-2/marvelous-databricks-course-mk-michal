import os

import databricks
import mlflow
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import AutoCaptureConfigInput, EndpointCoreConfigInput, ServedEntityInput

from hotel_cancellation.configs.config import Config
from hotel_cancellation.utils import get_logger

logger = get_logger()


class ModelServingEndpoint:
    def __init__(self, spark):
        self.registered_model_path = Config.REGISTERED_MODEL_NAME
        self.model_name = Config.REGISTERED_MODEL_NAME.split(".")[-1]
        self.workspace = WorkspaceClient()
        self.model_serving_endpoint = f"{self.model_name}-endpoint"
        self.spark = spark

    def get_latest_model_version(self):
        """
        Get the latest model version from the Databricks model registry.
        """
        client = mlflow.tracking.MlflowClient()
        logger.info(f"Getting the latest model version for {self.registered_model_path}")
        latest_version = client.get_model_version_by_alias(self.registered_model_path, alias="latest-model").version
        logger.info(f"Latest model version for {self.registered_model_path} is {latest_version}")
        return latest_version

    def deploy_serving_endpoint(self):
        """
        Deploys the model serving endpoint in Databricks.
        """
        latest_model_version = self.get_latest_model_version()

        # capture output in inference table
        auto_capture_config = AutoCaptureConfigInput(
            catalog_name=Config.CATALOG_NAME,
            enabled=True,
            schema_name=Config.SCHEMA_NAME,
            table_name_prefix=Config.TABLE_NAME_PREFIX,
        )

        try:
            current_config = self.workspace.serving_endpoints.get(name=self.model_serving_endpoint)
        except databricks.sdk.errors.platform.ResourceDoesNotExist:
            logger.info(f"Endpoint {self.model_serving_endpoint} does not exist. Creating endpoint")
            endpoint = self.workspace.serving_endpoints.create(
                name=self.model_serving_endpoint,
                config=EndpointCoreConfigInput(
                    auto_capture_config=auto_capture_config,
                    served_entities=[
                        ServedEntityInput(
                            entity_name=self.registered_model_path,
                            scale_to_zero_enabled=True,
                            workload_size="Small",
                            entity_version=latest_model_version,
                        )
                    ],
                ),
            )
            current_config = endpoint.config
        current_model_version = current_config.config.served_models[0].model_version
        if latest_model_version > current_model_version:
            logger.info(f"Updating endpoint from version {current_model_version} to {latest_model_version}")

            self.workspace.serving_endpoints.update_config(
                name=self.model_serving_endpoint,
                auto_capture_config=auto_capture_config,
                served_entities=[
                    ServedEntityInput(
                        entity_name=self.registered_model_path,
                        entity_version=latest_model_version,
                        scale_to_zero_enabled=current_config.config.served_entities[0].scale_to_zero_enabled,
                        workload_size=current_config.config.served_entities[0].workload_size,
                    )
                ],
            )
            logger.info(f"Endpoint {self.model_serving_endpoint} updated with latest model version")

    def call_endpoint(self, record: dict):
        """
        Calls the model serving endpoint with a given input record.
        """
        host = self.spark.conf.get("spark.databricks.workspaceUrl")
        serving_endpoint = f"https://{host}/serving-endpoints/{self.model_serving_endpoint}/invocations"

        logger.info(f"Calling endpoint: {serving_endpoint}")
        response = requests.post(
            serving_endpoint,
            headers={"Authorization": f"Bearer {os.environ['TOKEN']}"},
            json={"dataframe_records": [record]},
        )
        logger.info(f"Response from endpoint: {response.status_code}, {response.text}")
        if response.status_code != 200:
            logger.error(f"Error calling endpoint: {response.status_code}, {response.text}")
            response.raise_for_status()

        return response.status_code, response.text
