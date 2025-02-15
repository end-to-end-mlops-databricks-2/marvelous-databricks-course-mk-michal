import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from hotel_cancellation.configs.feature_pipeline_config import Config
from hotel_cancellation.model_training import ModelTrainer
from hotel_cancellation.utils import get_logger

logger = get_logger()


def create_feature_table(spark: SparkSession):
    logger.info(f"Creating feature table {Config.FEATURE_LOOKUP_CUSTOMER_TABLE}")
    sql = f"""
    CREATE OR REPLACE TABLE {Config.FEATURE_LOOKUP_CUSTOMER_TABLE}
    (customer_type STRING, previous_cancellations INT, previous_bookings_not_canceled INT, email STRING NOT NULL)
    """
    spark.sql(sql)
    spark.sql(f"ALTER TABLE {Config.FEATURE_LOOKUP_CUSTOMER_TABLE} ADD CONSTRAINT email_pk PRIMARY KEY (email)")
    spark.sql(
        f"ALTER TABLE {Config.FEATURE_LOOKUP_CUSTOMER_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )

    insert_sql = f"""
    INSERT INTO {Config.FEATURE_LOOKUP_CUSTOMER_TABLE}
        SELECT FIRST(customer_type) as customer_type, LAST(previous_cancellations) as previous_cancellations, LAST(previous_bookings_not_canceled) as previous_bookings_not_canceled, email
        FROM {Config.INPUT_TABLE}
        group by email
    """
    spark.sql(insert_sql)
    logger.info(f"Feature table {Config.FEATURE_LOOKUP_CUSTOMER_TABLE} created successfully")


class FeatureLookupTraining(ModelTrainer):
    def __init__(self, spark: SparkSession):
        self.workspace = WorkspaceClient()

        self.workspace_id = spark.conf.get("spark.databricks.workspaceUrl").split(".")[0]
        self.spark = spark
        self.fe = feature_engineering.FeatureEngineeringClient()
        self.target = "is_canceled"
        self.feature_lookup_columns = ["customer_type", "previous_cancellations", "previous_bookings_not_canceled"]
        self.feature_lookup_key = "email"
        self.model_pipeline = None

    def feature_engineering(self):
        self.training_set = self.fe.create_training_set(
            df=self.train_df,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=Config.FEATURE_LOOKUP_CUSTOMER_TABLE,
                    feature_names=self.feature_lookup_columns,
                    lookup_key=self.feature_lookup_key,
                ),
            ],
        )
        self.X_train = self.training_set.load_df().toPandas()
        self.y_train = self.train_df.toPandas()[self.target]

        self.y_test = self.test_df[self.target]
        self.X_test = self.test_df.drop(self.target, axis=1)

    def load_data_and_drop_columns(self):
        # Combine the columns to drop into a single list
        columns_to_drop = self.feature_lookup_columns
        self.train_df = self.spark.table(Config.OUTPUT_TRAIN_TABLE).drop(*columns_to_drop)
        self.test_df = self.spark.table(Config.OUTPUT_TEST_TABLE).toPandas()

    def log_results_to_mlflow(self, metrics: dict):
        mlflow.set_experiment("hotel_cancellation_feature_lookup")
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.log_params(Config.LOGISTIC_REGRESSION_PARAMETERS)

            dataset = mlflow.data.from_spark(
                self.train_df,
                table_name=Config.OUTPUT_TRAIN_TABLE,
                version="0",
            )
            mlflow.log_input(dataset, context="training")

            # create signature
            signature = infer_signature(self.X_train, self.y_train)

            self.fe.log_model(
                model=self.model_pipeline,
                flavor=mlflow.sklearn,
                artifact_path="model-logistic-regression-feature-lookup",
                training_set=self.training_set,
                signature=signature,
            )

            
