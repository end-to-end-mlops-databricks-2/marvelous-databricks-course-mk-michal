import os

import mlflow

from hotel_cancellation.serving.endpoint_creation_basic import ModelServingEndpoint

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

os.environ["TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

model_serving = ModelServingEndpoint(spark)

model_serving.deploy_serving_endpoint()


record = {
    "lead_time": 10.0,
    "total_of_special_requests": 1.0,
    "required_car_parking_spaces": 1.0,
    "booking_changes": 1.0,
    "previous_cancellations": 1.0,
    "previous_bookings_not_canceled": 0.0,
    "adr": 100.0,
    "stays_in_week_nights": 2.0,
    "is_repeated_guest": 1.0,
    "agent": 0,
    "company": 0,
    "hotel": "",
    "arrival_date_month": "January",
    "meal": "BB",
    "market_segment": "Direct",
    "distribution_channel": "Direct",
    "reserved_room_type": "A",
    "customer_type": "Transient",
    "email": "",
}


model_serving.call_endpoint(record)
