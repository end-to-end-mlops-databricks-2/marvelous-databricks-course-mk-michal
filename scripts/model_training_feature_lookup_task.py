# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------


import sys
sys.path
sys.path.append('../src')

from hotel_cancellation.model_training_with_feature_lookup import create_feature_table, FeatureLookupTraining
import mlflow


# COMMAND ----------


import mlflow
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
feature_lookup_training = FeatureLookupTraining(spark)
feature_lookup_training.load_data_and_drop_columns()

# COMMAND ----------

feature_lookup_training.feature_engineering()

# COMMAND ----------


feature_lookup_training.create_pipeline()
trained_model = feature_lookup_training.fit()

# COMMAND ----------

metrics = feature_lookup_training.evaluate()
feature_lookup_training.log_results_to_mlflow(metrics)

# COMMAND ----------


