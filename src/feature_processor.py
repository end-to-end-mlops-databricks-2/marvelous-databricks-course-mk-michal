import base64
import io
import os

import joblib
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from configs.feature_pipeline_config import Config
from src.utils import get_logger

logger = get_logger()

class BinaryThresholdTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for binarizing columns (values > 0 become 1 for numeric, non-empty strings become 1)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if pd.api.types.is_numeric_dtype(X):
            return (X > 0).astype(int)
        else:
            # For string columns, return 1 for non-empty strings (excluding missing/NaN)
            return (~pd.isna(X) & (X != '')).astype(int)


def combine_feature_names(x, y):
    """Combines feature names for OneHotEncoder by joining them with underscore and replacing spaces"""
    return f"{x}_{y}".replace(' ', '_')

def create_preprocessing_pipeline(config):
    """Create a sklearn pipeline based on the configuration"""

    # Separate columns by transformation type
    numerical_columns = []
    categorical_columns = []
    binary_columns = []

    for col in config:
        if col['transformation'] == 'one-hot':
            categorical_columns.append(col['column'])
        elif col['transformation'] == 'binarize':
            binary_columns.append(col['column'])
        elif col['transformation'] is None:
            numerical_columns.append(col['column'])

    # Create transformers for each column type
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore',
                                feature_name_combiner=combine_feature_names))
    ])

    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('binarizer', BinaryThresholdTransformer())
    ])

    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns),
            ('bin', binary_transformer, binary_columns)
        ],
        remainder='drop'  # Drop any columns not specified in the transformers
    )

    return preprocessor


class HotelBookingPreprocessor:
    """Main preprocessor class that handles the entire pipeline"""

    def __init__(self, config):
        self.config = config
        self.pipeline = create_preprocessing_pipeline(config)
        self.feature_names = None

    def fit(self, X, y=None):
        """Fit the pipeline and store feature names"""
        self.pipeline.fit(X)

        # Get feature names for all transformers
        feature_names = []

        # Get column transformer
        column_transformer = self.pipeline

        # Get feature names for numerical columns
        numerical_columns = [col['column'] for col in self.config if col['transformation'] is None]
        feature_names.extend(numerical_columns)

        # Get feature names for categorical columns
        categorical_columns = [col['column'] for col in self.config if col['transformation'] == 'one-hot']
        categorical_features = column_transformer.named_transformers_['cat'].named_steps[
            'onehot'].get_feature_names_out(categorical_columns)
        feature_names.extend(categorical_features)

        # Get feature names for binary columns
        binary_columns = [col['column'] for col in self.config if col['transformation'] == 'binarize']
        feature_names.extend(binary_columns)

        self.feature_names = feature_names
        return self

    def transform(self, X):
        """Transform the data and return a DataFrame with proper column names"""
        if self.feature_names is None:
            raise ValueError("Pipeline must be fitted before calling transform")

        transformed_array = self.pipeline.transform(X)
        return pd.DataFrame(transformed_array, columns=self.feature_names)

    def fit_transform(self, X, y=None):
        """Fit and transform the data"""
        return self.fit(X, y).transform(X)

    def save(self, transformed_data: pd.DataFrame, spark, dbutils):
        """Save preprocessing pipeline and transformed data to Databricks Volumes"""

        run_id = int(pd.Timestamp.now().timestamp())

        # Define paths for Volumes (without /Volumes prefix as dbutils will handle it)
        volume_base_path = os.path.join(Config.OUTPUT_PREPROCESSING, str(run_id))
        pipeline_path = os.path.join(volume_base_path, "pipeline.joblib")
        config_path = os.path.join(volume_base_path, "columns_config.yaml")

        dbutils.fs.mkdirs(volume_base_path)
        logger.info(f"Making new directory: {volume_base_path}")

        dbutils.fs.mkdirs(volume_base_path)

        # Serialize pipeline to bytes and save using dbutils

        # Convert to spark dataframe and save
        spark_df = spark.createDataFrame(transformed_data)
        (spark_df.write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(Config.OUTPUT_PROCESSED_DATA_PATH))
        logger.info(f"Processed data saved to Delta table: {Config.OUTPUT_PROCESSED_DATA_PATH}")

        # Save metadata
        metadata = {
            "processing_date": pd.Timestamp.now().isoformat(),
            "feature_count": len(transformed_data.columns),
            "row_count": len(transformed_data),
            "run_id": run_id,
            "pipeline_path": pipeline_path,
            "config_path": config_path
        }

        (spark.createDataFrame([metadata])
            .write
            .format("delta")
            .mode("append")
            .saveAsTable(Config.OUTPUT_PROCESSED_DATA_PATH_METADATA))
        logger.info(f"Metadata saved successfully into {Config.OUTPUT_PROCESSED_DATA_PATH_METADATA}")

        # Save config with features and transformations to DBFS
        config_data = {
            'columns_config': Config.COLUMNS_CONFIG,
            'feature_names': self.feature_names,
            'run_id': run_id
        }
        yaml_content = yaml.dump(config_data)
        # Save to DBFS using dbutils
        dbutils.fs.put(config_path, yaml_content, overwrite=True)
        logger.info(f"Config saved to: {config_path}")


        # save pipeline.joblib into databricks volume using dbutils
        buffer = io.BytesIO()
        joblib.dump(self.pipeline, buffer)
        encoded_content = base64.b64encode(buffer.getvalue()).decode('utf-8')
        dbutils.fs.put(pipeline_path, encoded_content, overwrite=True)
        logger.info(f"Pipeline saved to: {pipeline_path}")

        return {
            "run_id": run_id,
            "table_name": Config.OUTPUT_PROCESSED_DATA_PATH,
            "pipeline_path": pipeline_path,
            "config_path": config_path
        }

