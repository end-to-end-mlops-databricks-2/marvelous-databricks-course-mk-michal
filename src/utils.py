from flask import logging


def get_logger(level=logging.INFO):
	logger = logging.getLogger("default_logger")
	if not logger.hasHandlers():  # Ensure no duplicate handlers
		handler = logging.StreamHandler()
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		logger.addHandler(handler)
	logger.setLevel(level)
	return logger