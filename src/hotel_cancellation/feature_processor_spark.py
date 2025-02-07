from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, NumericType

from src.hotel_cancellation.utils import get_logger

logger = get_logger()


class BinaryThresholdTransformer:
    """Custom transformer for binarizing columns (values > 0 become 1 for numeric, non-empty strings become 1)"""

    @staticmethod
    def transform_numeric(column_name):
        """Transform numeric columns to binary (1 if > 0, else 0)"""
        return F.when(F.col(column_name) > 0, 1).otherwise(0)

    @staticmethod
    def transform_string(column_name):
        """Transform string columns to binary (1 if non-empty and non-null, else 0)"""
        return F.when(F.col(column_name).isNotNull() & (F.length(F.trim(F.col(column_name))) > 0), 1).otherwise(0)


class HotelBookingPreprocessorSpark:
    """Main preprocessor class that handles the Spark-based preprocessing pipeline"""

    def __init__(self, config):
        self.config = config
        self.feature_names = None
        self.label_column = "is_canceled"
        self._separate_columns()

    def _separate_columns(self):
        """Separate columns by transformation type"""
        self.numerical_columns = []
        self.binary_columns = []

        for col in self.config:
            if col["transformation"] == "binarize":
                self.binary_columns.append(col["column"])
            elif col["transformation"] is None:
                self.numerical_columns.append(col["column"])

    def _handle_numeric_columns(self, df: DataFrame) -> DataFrame:
        """Handle numeric columns: impute with mean"""
        for col in self.numerical_columns:
            logger.info(f"Processing column {col} of type {df.schema[col].dataType}")
            # Calculate mean for imputation
            mean_value = df.select(F.mean(col)).collect()[0][0]

            # Impute nulls with mean and cast to double
            df = df.withColumn(col, F.coalesce(F.col(col).cast(DoubleType()), F.lit(mean_value)))

        return df

    def _handle_binary_columns(self, df: DataFrame) -> DataFrame:
        """Handle binary columns: impute with 0 and apply binary transformation"""
        for col in self.binary_columns:
            logger.info(f"Processing column {col} of type {df.schema[col].dataType}")
            # Check if column type is numeric using isinstance
            is_numeric = isinstance(df.schema[col].dataType, NumericType)

            # Apply appropriate transformation
            if is_numeric:
                df = df.withColumn(col, BinaryThresholdTransformer.transform_numeric(col))
            else:
                df = df.withColumn(col, BinaryThresholdTransformer.transform_string(col))

        return df

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform the data using Spark transformations"""
        # Process numeric columns
        df = self._handle_numeric_columns(df)

        # Process binary columns
        df = self._handle_binary_columns(df)

        # Select only the columns we processed
        columns_to_keep = self.numerical_columns + self.binary_columns + [self.label_column]

        return df.select(columns_to_keep)
