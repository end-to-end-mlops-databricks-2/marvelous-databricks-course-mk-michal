import logging
import os
import pandas as pd
import yaml
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from configs.feature_pipeline_config import Config
from src.utils import get_logger

logger = get_logger()

class BinaryThresholdTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for binarizing columns (values > 0 become 1)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X > 0).astype(int)


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
        ('onehot', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'))
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

    def save(self, transformed_data: pd.DataFrame):
        """Save preprocessing pipeline, transformed data, columns info as artifacts"""
        run_id = int(pd.Timestamp.now().timestamp())

        output_path = os.path.join(Config.OUTPUT_DATA_PATH, str(run_id))
        os.makedirs(output_path, exist_ok=True)
        logger.info(f'Saving files into folder: {output_path}')

        dump(self.pipeline, os.path.join(output_path, Config.OUTPUT_PIPELINE))

        transformed_data.to_csv(os.path.join(output_path, Config.OUTPUT_DATA_PATH_FEATURES), index=False)

        # # save category mapping as json
        # with open(os.path.join(output_path, Config.OUTPUT_DATA_PATH_CATEGORY_MAPPING), 'w') as f:
        #     yaml.dump({'category_mapping': Config._category_columns_mapping}, f, indent=4)

        # save config file with features used and transformations as yaml


        with open(os.path.join(output_path, Config.OUTPUT_DATA_PATH_COLUMNS), 'w') as f:
            yaml.dump(
                {
                    'columns_config': Config.COLUMNS_CONFIG,
                    'feature_names': self.feature_names
                 }, f
            )

        return output_path

        # save model params as yaml


    @classmethod
    def load(cls, path):
        """Load a saved preprocessor"""
        return load(path)