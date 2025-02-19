import mlflow
import pandas as pd
import pyspark
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_cancellation.configs.feature_pipeline_config import Config
from hotel_cancellation.utils import get_logger

logger = get_logger()


class ModelTrainer:
    def __init__(
        self,
        train_data: pyspark.sql.DataFrame,
        test_data: pyspark.sql.DataFrame,
    ):
        self.train_data = train_data
        self.test_data = test_data

        self.model_pipeline = None
        self.X_train, self.y_train = self.prepare_data(self.train_data.toPandas())
        self.X_test, self.y_test = self.prepare_data(self.test_data.toPandas())

    def prepare_data(self, data: pd.DataFrame):
        """Return X and y from the input training data"""
        y = data["is_canceled"]
        X = data.drop("is_canceled", axis=1)
        logger.info(f"Train data have shape: {X.shape}")
        return X, y

    def create_pipeline(self):
        """Create a pipeline that combines preprocessing and model training"""
        # Get categorical and numerical columns from config
        categorical_columns = [
            col["column"]
            for col in Config.COLUMNS_CONFIG
            if col["type"] == "category" and col["transformation"] == "one-hot"
        ]

        numerical_columns = [
            col["column"]
            for col in Config.COLUMNS_CONFIG
            if col["type"] != "category" or col["transformation"] != "one-hot"
        ]

        # Create preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numerical_columns),
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_columns),
            ]
        )

        # Create the full pipeline
        self.model_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(**Config.LOGISTIC_REGRESSION_PARAMETERS),
                ),  # Replace with your preferred model
            ]
        )
        logger.info(f"Pipeline created: {self.model_pipeline.named_steps}")

    def fit(self):
        """Fit the pipeline on the training data"""
        if self.model_pipeline is None:
            self.create_pipeline()

        self.model_pipeline.fit(self.X_train, self.y_train)
        return self.model_pipeline

    def evaluate(self):
        """Evaluate the pipeline on the test data"""
        if self.model_pipeline is None:
            self.create_pipeline()

        y_pred = self.model_pipeline.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        return {"accuracy": accuracy, "f1": f1}

    def log_results_to_mlflow(self, metrics: dict):
        mlflow.log_metrics(metrics)
        mlflow.log_params(Config.LOGISTIC_REGRESSION_PARAMETERS)

        dataset = mlflow.data.from_spark(
            self.train_data,
            table_name=Config.OUTPUT_TRAIN_TABLE,
            version="0",
        )
        mlflow.log_input(dataset, context="training")

        dataset_test = mlflow.data.from_spark(self.test_data, table_name=Config.OUTPUT_TEST_TABLE, version="0")
        mlflow.log_input(dataset_test, context="training")

        # create signature
        signature = infer_signature(self.X_train, self.y_train)

        mlflow.sklearn.log_model(
            sk_model=self.model_pipeline,
            artifact_path="model-logistic-regression",
            signature=signature,
        )
