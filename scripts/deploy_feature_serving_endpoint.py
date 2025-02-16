# Databricks notebook source
from hotel_cancellation.serving.feature_serving_endpoint import FeatureServingEndpoint


feature_serving = FeatureServingEndpoint(spark)
feature_serving.create_online_feature_table()

# COMMAND ----------
# Create feature spec
feature_serving.create_feature_spec()

# COMMAND ----------
# Deploy feature serving endpoint
feature_serving.deploy_or_update_serving_endpoint()