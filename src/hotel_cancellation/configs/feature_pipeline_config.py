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

    CATALOG_NAME = "mlops_dev"
    SCHEMA_NAME = "michalku"
    TABLE_NAME_PREFIX = "hotel_bookings_inference"


    INPUT_TABLE = "mlops_dev.michalku.hotel_bookings_full"
    OUTPUT_TRAIN_TABLE = "mlops_dev.michalku.hotel_bookings_train"
    OUTPUT_TEST_TABLE = "mlops_dev.michalku.hotel_bookings_test"

    # training parameters
    LOGISTIC_REGRESSION_PARAMETERS = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "liblinear",
        "max_iter": 100,
    }

    REGISTERED_MODEL_NAME = "mlops_dev.michalku.hotel_cancellation_model_log_r"
    REGISTERED_MODEL_NAME_FE = "mlops_dev.michalku.hotel_cancellation_model_fe"
    MAIN_MODEL_METRIC = "f1"
    EXPERIMENT_NAME = "/Shared/hotel-cancellation"

    FEATURE_LOOKUP_CUSTOMER_TABLE = "mlops_dev.michalku.feature_table_customer"
