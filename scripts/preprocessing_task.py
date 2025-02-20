from hotel_cancellation.configs.config import Config
from hotel_cancellation.feature_processor_spark import HotelBookingPreprocessorSpark
from hotel_cancellation.utils import get_logger

logger = get_logger()

df = spark.table(Config.INPUT_TABLE)
preprocessor = HotelBookingPreprocessorSpark(Config.COLUMNS_CONFIG)
transformed_df = preprocessor.transform(df)

# display the first 10 rows of the transformed dataframe

# split the dataframe into train and test
logger.info("Splitting the dataframe into train and test")
train_df, test_df = transformed_df.randomSplit([0.8, 0.2], seed=42)

# Drop existing tables if they exist
logger.info("Dropping existing tables if they exist")
spark.sql(f"DROP TABLE IF EXISTS {Config.OUTPUT_TRAIN_TABLE}")
spark.sql(f"DROP TABLE IF EXISTS {Config.OUTPUT_TEST_TABLE}")

# save the train and test dataframes to delta tables
logger.info("Saving the train and test dataframes to delta tables")
train_df.write.format("delta").mode("overwrite").saveAsTable(Config.OUTPUT_TRAIN_TABLE)
test_df.write.format("delta").mode("overwrite").saveAsTable(Config.OUTPUT_TEST_TABLE)
