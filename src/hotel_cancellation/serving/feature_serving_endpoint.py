from hotel_cancellation.configs.feature_pipeline_config import Config

import databricks
from databricks.feature_engineering import FeatureLookup
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from databricks.feature_engineering import FeatureEngineeringClient

from hotel_cancellation.utils import get_logger

logger = get_logger()

class FeatureServingEndpoint:
    def __init__(self, spark):
        self.feature_lookup_table = Config.FEATURE_LOOKUP_CUSTOMER_TABLE
        self.feature_lookup_table_online = Config.FEATURE_LOOKUP_CUSTOMER_TABLE_ONLINE
        self.feature_spec_name = Config.FEATURE_SPEC_NAME
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = FeatureEngineeringClient()

    def create_online_feature_table(self):
        """
        Creates an online feature table in Databricks if it doesn't already exist.
        """
        # Check if table exists
        try:
            self.workspace.online_tables.get(self.feature_lookup_table_online)
            logger.info(f"Online feature table {self.feature_lookup_table_online} already exists. Skipping creation.")
        except databricks.sdk.errors.platform.NotFound:
            logger.info(f"Online feature table {self.feature_lookup_table_online} does not exist. Creating it.")
        
            spec = OnlineTableSpec(
                primary_key_columns=["email"],
                source_table_full_name=self.feature_lookup_table,
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
                perform_full_copy=False,
            )
            logger.info(f"Creating online feature table {self.feature_lookup_table_online} with spec {spec}")

            self.workspace.online_tables.create(name=self.feature_lookup_table_online, spec=spec)
            logger.info(f"Online feature table {self.feature_lookup_table_online} created")



    def create_feature_spec(self):
        """
        Creates a feature spec in Databricks.
        """
        features = [
            FeatureLookup(
                table_name=self.feature_lookup_table,
                lookup_key="email_id",
                feature_names=["customer_type", "previous_cancellations", "previous_bookings_not_cancelled"],
            )
        ]
        logger.info(f"Creating feature spec {self.feature_spec_name} with features {features}")
        self.fe.create_feature_spec(name=self.feature_spec_name, features=features, exclude_columns=None)
        logger.info(f"Feature spec {self.feature_spec_name} created")

    def deploy_or_update_serving_endpoint(self, workload_size: str = "Small", scale_to_zero: bool = True):
        """
        Deploys the feature serving endpoint in Databricks.
        :param workload_seze: str. Workload size (number of concurrent requests). Default is Small = 4 concurrent requests
        :param scale_to_zero: bool. If True, endpoint scales to 0 when unused.
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        logger.info(f"Deploying or updating feature serving endpoint {Config.FEATURE_LOOKUP_ENDPOINT_NAME}")
        served_entities = [
            ServedEntityInput(
                entity_name=self.feature_spec_name, scale_to_zero_enabled=scale_to_zero, workload_size=workload_size
            )
        ]
        if not endpoint_exists:
            logger.info(f"Creating feature serving endpoint {Config.FEATURE_LOOKUP_ENDPOINT_NAME} with served entities {served_entities}")
            self.workspace.serving_endpoints.create(
                name=Config.FEATURE_LOOKUP_ENDPOINT_NAME,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            logger.info(f"Updating feature serving endpoint {Config.FEATURE_LOOKUP_ENDPOINT_NAME} with served entities {served_entities}")
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)
