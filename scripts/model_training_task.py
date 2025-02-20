import mlflow
import mlflow.utils.databricks_utils
from pyspark.sql import SparkSession

from hotel_cancellation.configs.config import Config
from hotel_cancellation.model_training import ModelTrainer

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
spark = SparkSession.builder.getOrCreate()

print(spark.version)


mlflow.set_experiment(Config.EXPERIMENT_NAME)
# Start MLflow run
with mlflow.start_run() as run:
    # Load and prepare data
    train_data = spark.table(Config.OUTPUT_TRAIN_TABLE)
    test_data = spark.table(Config.OUTPUT_TEST_TABLE)

    trainer = ModelTrainer(train_data, test_data)
    trainer.create_pipeline()
    # Train the model
    trained_model = trainer.fit()
    metrics = trainer.evaluate()

    # Log parameters from config

    # Evaluate and log metrics
    metrics = trainer.evaluate()
    trainer.log_results_to_mlflow(metrics)

# COMMAND ----------
