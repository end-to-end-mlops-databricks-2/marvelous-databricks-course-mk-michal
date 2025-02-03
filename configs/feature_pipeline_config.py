class Config:
    COLUMNS_CONFIG = [
        {"column": "lead_time", "type": "int", "transformation": None, "fill_missing": "mean"},
        {"column": "total_of_special_requests", "type": "float", "transformation": None, "fill_missing": 0},
        {"column": "required_car_parking_spaces", "type": "float", "transformation": None, "fill_missing": 0},
        {"column": "booking_changes", "type": "float", "transformation": None, "fill_missing": 0},
        {"column": "previous_cancellations", "type": "float", "transformation": None, "fill_missing": 0},
        {"column": "is_repeated_guest", "type": "float", "transformation": None, "fill_missing": 0},
        {"column": "previous_bookings_not_canceled", "type": "float", "transformation": None, "fill_missing": 0},
        {"column": "adr", "type": "float", "transformation": None, "fill_missing": "mean"},
        {"column": "stays_in_week_nights", "type": "float", "transformation": None, "fill_missing": "mean"},
        {"column": "hotel", "type": "category", "transformation": "one-hot", "fill_missing": 0},
        {"column": "agent", "type": "category", "transformation": "binarize", "fill_missing": 0},
        {"column": "company", "type": "category", "transformation": "binarize", "fill_missing": 0},
        {"column": "arrival_date_month", "type": "category", "transformation": "one-hot", "fill_missing": None},
        {"column": "meal", "type": "category", "transformation": "one-hot", "fill_missing": None},
        {"column": "market_segment", "type": "category", "transformation": "one-hot", "fill_missing": None},
        {"column": "distribution_channel", "type": "category", "transformation": "one-hot", "fill_missing": None},
        {"column": "reserved_room_type", "type": "category", "transformation": "one-hot", "fill_missing": None},
        {"column": "customer_type", "type": "category", "transformation": "one-hot", "fill_missing": None},
    ]

    OUTPUT_PROCESSED_DATA_PATH = "mlops_dev.michalku.hotel_bookings_processed"
    OUTPUT_PROCESSED_DATA_PATH_METADATA = f"{OUTPUT_PROCESSED_DATA_PATH}_metadata"

    OUTPUT_PREPROCESSING = "/Volumes/mlops_dev/michalku/preprocessing"
