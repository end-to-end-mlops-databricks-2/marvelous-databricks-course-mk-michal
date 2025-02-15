import os
import mlflow
from hotel_cancellation.serving.endpoint_creation_basic import ModelServingEndpoint

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

os.environ["TOKEN"] = "eyJraWQiOiJkZmJjOWVmMThjZTQ2ZTlhMDg2NWZmYzlkODkxYzJmMjg2NmFjMDM3MWZiNDlmOTdhMDg1MzBjNWYyODU3ZTg4IiwidHlwIjoiYXQrand0IiwiYWxnIjoiUlMyNTYifQ.eyJjbGllbnRfaWQiOiJkYXRhYnJpY2tzLWNsaSIsInNjb3BlIjoib2ZmbGluZV9hY2Nlc3MgYWxsLWFwaXMiLCJpc3MiOiJodHRwczovL2RiYy00ODk0MjMyYi05ZmM1LmNsb3VkLmRhdGFicmlja3MuY29tL29pZGMiLCJhdWQiOiIzNzM0NTQ3MzQ0NTA1OTYxIiwic3ViIjoibWljaGFsa3VjaXJrYUBnbWFpbC5jb20iLCJpYXQiOjE3Mzk0NTM2NjMsImV4cCI6MTczOTQ1NzI2MywianRpIjoiYjU3MTk5ZWItYzJiMy00MmMzLTgxM2UtZDk2NTM0ODQwZTZjIn0.dl2EBiZCn7urd8V01cEXb6bcWzROmu1i7oeCwq-h56unmSp-a4zAd6ziZmJXFT8BzTyOw76ax7O32x1CjmFW-E-Tuf2jfLN1fbKObvLB7JoQ-5HdQ4_Orl4LSWEpOEvQDvfgkij6NV_whrjKJMuQUEplndJWx-W8AI2ZJ2DG0g0BUaNsGUOnJ0L3VV4y2socVFkBCwrJZdRtOdPvhwLMk_m27-wc34jY2YkmDoGoYtF5-Pu5edvBlGG5N9mp8GzLIcSHdy4QZHGMMrPPJe5RGbVlSwo0-bepuEyyR9j4Zra65wi9nZc1IFJZo9EepPb_T7noh4j_YvFn_uWiIGr74A"
# os.environ["TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
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
    "email": ""
    }


model_serving.call_endpoint(record)