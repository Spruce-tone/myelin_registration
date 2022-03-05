import logging

import logging.config
import json
import os

if not os.path.isdir('./logs'):
    os.makedirs('./logs', exist_ok=True)
        
LOGGING_CONFIG = {
    'version': 1,
    'loggers': {
        'info': {  
            'level': 'DEBUG',
            'handlers': ['info_console_handler', 'debug_rotating_file_handler', 'error_file_handler'],
        },
    },
    
    'handlers': {
        'info_console_handler': {
            'level': 'INFO',
            'formatter': 'info_format',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },

        'debug_rotating_file_handler': {
            'level': 'DEBUG',
            'formatter': 'debug_format',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': './logs/info.log',
            'mode': 'w',
            'maxBytes': 1048576,
            'backupCount': 10
        },

        'error_file_handler': {
            'level': 'ERROR',
            'formatter': 'debug_format',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': './logs/error.log',
            'mode': 'w',
            'maxBytes': 1048576,
            'backupCount': 10
        },

    },

    'formatters': {
        'info_format': {
            'format': '[%(levelname)s | %(asctime)s | %(filename)s | %(funcName)s] (line %(lineno)d) %(message)s :: %(name)s logger'
        },
        'debug_format': {
            'format': '[%(levelname)s | %(asctime)s| %(module)s | %(filename)s | %(funcName)s]\n (line %(lineno)d) %(message)s :: %(name)s logger \n'
        },
    },

}

class CustomLogger():
    def __init__(self):
        self.LOGGER_CONFIG_PATH = "./lib/logger_config.json"
        self.LOG_FILE_PATH = "./logs"
        if os.path.isdir(self.LOG_FILE_PATH):
            os.makedirs(self.LOG_FILE_PATH, exist_ok=True)
        
        self.info_logger = self._get_logger('info')

    def _get_logger(self, logger_name: str):
        """
        make logger instance from logger config file

        ------------------------------
        Input Args:
            logger_name : str
                enter logger name
        """
        logging.config.dictConfig(LOGGING_CONFIG)
        return logging.getLogger(logger_name)
