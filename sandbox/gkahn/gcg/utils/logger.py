import os
import logging
from colorlog import ColoredFormatter

_color_formatter = ColoredFormatter(
    "%(asctime)s %(log_color)s%(name)-10s %(levelname)-8s%(reset)s %(white)s%(message)s",
    datefmt='%m-%d %H:%M:%S',
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

_normal_formatter = logging.Formatter(
    '%(asctime)s %(name)-10s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S',
    style='%'
)

DEBUG_LOG_PATH = '/tmp/log.txt'
def _set_log_path(path):
    global DEBUG_LOG_PATH
    DEBUG_LOG_PATH = path

_LOGGERS = {}
def _get_logger(name, lvl=logging.INFO, log_path=DEBUG_LOG_PATH, display_name=None):
    global _LOGGERS

    if isinstance(lvl, str):
        lvl = lvl.lower().strip()
        if lvl == 'debug': lvl = logging.DEBUG
        elif lvl == 'info': lvl = logging.INFO
        elif lvl == 'warn' or lvl == 'warning': lvl = logging.WARN
        elif lvl == 'error': lvl = logging.ERROR
        elif lvl == 'fatal' or lvl == 'critical': lvl = logging.CRITICAL
        else: raise ValueError('unknown logging level')

    logger = _LOGGERS.get(name, None)
    if logger is not None: return logger

    # file_handler = logging.FileHandler(GET_FILE_MAN().debug_log_path)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_normal_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(lvl)
    console_handler.setFormatter(_color_formatter)
    if display_name is None:
        display_name = name
    logger = logging.getLogger(display_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    _LOGGERS[name] = logger
    return logger

GLOBAL_LOGGER_NAME = '_global_logger'
def setup_logger(log_path, lvl):
    display_name = os.path.dirname(log_path).split('/')[-1]
    _get_logger(GLOBAL_LOGGER_NAME, lvl=lvl, log_path=log_path, display_name=display_name)

def debug(s):
    assert (GLOBAL_LOGGER_NAME in _LOGGERS)
    _LOGGERS[GLOBAL_LOGGER_NAME].debug(s)

def info(s):
    assert (GLOBAL_LOGGER_NAME in _LOGGERS)
    _LOGGERS[GLOBAL_LOGGER_NAME].info(s)

def warn(s):
    assert (GLOBAL_LOGGER_NAME in _LOGGERS)
    _LOGGERS[GLOBAL_LOGGER_NAME].warn(s)

def error(s):
    assert (GLOBAL_LOGGER_NAME in _LOGGERS)
    _LOGGERS[GLOBAL_LOGGER_NAME].error(s)

def critical(s):
    assert (GLOBAL_LOGGER_NAME in _LOGGERS)
    _LOGGERS[GLOBAL_LOGGER_NAME].critical(s)
