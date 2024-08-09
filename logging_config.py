import logging
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler
import os


def setup_logging():
    logger = logging.getLogger()
    if not logger.handlers:  # Check if handlers are already configured
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Default to INFO, can be set via environment variable
        log_format = '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        log_dir = "logs"
        log_file = "application.log"

        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Configure the logging settings
        dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': log_format,
                },
            },
            'handlers': {
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': os.path.join(log_dir, log_file),
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'formatter': 'default',
                    'level': log_level,
                },
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                    'level': log_level,
                },
            },
            'root': {
                'level': log_level,
                'handlers': ['file', 'console'],
            },
        })
