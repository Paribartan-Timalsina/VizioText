from parameters import DEBUG, LOGGER_NAME
import logging

logger = None

def get_logger():
    global logger
    if logger is None:
        # setting up the logger
        logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, force=True) # a workaround
        logger = logging.getLogger(LOGGER_NAME)
        return logger
    else:
        return logger
