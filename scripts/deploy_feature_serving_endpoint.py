# Databricks notebook source
# MAGIC %pip install uv
# MAGIC

# COMMAND ----------

# !uv pip install  ..
# %restart_python

# COMMAND ----------

import json
import os

import pandas as pd
import requests

from hotel_cancellation.serving.feature_serving_endpoint import FeatureServingEndpoint

feature_serving = FeatureServingEndpoint(spark)
feature_serving.create_online_feature_table()
feature_serving.create_feature_spec()
feature_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------


os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

emails_spark_df = spark.table("mlops_dev.michalku.feature_table_customer").select("email").limit(4)


def create_tf_serving_json(data):
    return {"inputs": {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}


def score_model(dataset):
    url = "https://dbc-4894232b-9fc5.cloud.databricks.com/serving-endpoints/hotel_cancellation_feature_lookup-endpoint/invocations"
    headers = {"Authorization": f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', "Content-Type": "application/json"}
    ds_dict = (
        {"dataframe_split": dataset.to_dict(orient="split")}
        if isinstance(dataset, pd.DataFrame)
        else create_tf_serving_json(dataset)
    )
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()


score_model(emails_spark_df.toPandas())
