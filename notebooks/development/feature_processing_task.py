# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Processing Task
# MAGIC This notebook contains code for processing features.


# COMMAND ----------
from configs.feature_pipeline_config import Config
from src.feature_processor import HotelBookingPreprocessor

df = spark.table('mlops_dev.michalku.hotel_bookings')
df_pandas = df.toPandas()
preprocessor = HotelBookingPreprocessor(Config.COLUMNS_CONFIG)
transformed_df = preprocessor.fit_transform(df_pandas)
transformed_df['is_canceled'] = df_pandas['is_canceled']

# COMMAND ----------
preprocessor.save(transformed_df, spark, dbutils)