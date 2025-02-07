import logging


def get_logger(level=logging.INFO):
    logger = logging.getLogger("default_logger")
    # Prevent log messages from propagating to the root logger
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
    handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
