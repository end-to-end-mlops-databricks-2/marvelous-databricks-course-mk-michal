# Databricks notebook source

# set token 
# import os
# os.environ["DATABRICKS_TOKEN"] = "eyJraWQiOiJkZmJjOWVmMThjZTQ2ZTlhMDg2NWZmYzlkODkxYzJmMjg2NmFjMDM3MWZiNDlmOTdhMDg1MzBjNWYyODU3"

import mlflow
from hotel_cancellation.model_training_with_feature_lookup import FeatureLookupTraining, create_feature_table
# create_feature_table(spark)

# COMMAND ----------
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
feature_lookup_training = FeatureLookupTraining(spark)
feature_lookup_training.load_data_and_drop_columns()
feature_lookup_training.feature_engineering()
feature_lookup_training.train()
feature_lookup_training.evaluate()
feature_lookup_training.log_results_to_mlflow()
# COMMAND ----------