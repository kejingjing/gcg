import os, csv
import logging
from colorlog import ColoredFormatter

from .tabulate import tabulate

class LoggerClass(object):
    GLOBAL_LOGGER_NAME = '_global_logger'

    _color_formatter = ColoredFormatter(
        "%(asctime)s %(log_color)s%(name)-10s %(levelname)-8s%(reset)s %(white)s%(message)s",
        datefmt='%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
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
    
    def __init__(self):
        self._logger = None
        self._log_path = None
        self._csv_path = None
        self._tabular = list()
        
    #############
    ### Setup ###
    #############
        
    def setup(self, display_name, log_path, lvl):
        self._logger = self._get_logger(LoggerClass.GLOBAL_LOGGER_NAME,
                                        log_path,
                                        lvl=lvl,
                                        display_name=display_name)
        self._csv_path = os.path.splitext(log_path)[0] + '.csv'
        self._tabular_keys = None

    def _get_logger(self, name, log_path, lvl=logging.INFO, display_name=None):
        if isinstance(lvl, str):
            lvl = lvl.lower().strip()
            if lvl == 'debug':
                lvl = logging.DEBUG
            elif lvl == 'info':
                lvl = logging.INFO
            elif lvl == 'warn' or lvl == 'warning':
                lvl = logging.WARN
            elif lvl == 'error':
                lvl = logging.ERROR
            elif lvl == 'fatal' or lvl == 'critical':
                lvl = logging.CRITICAL
            else:
                raise ValueError('unknown logging level')

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(LoggerClass._normal_formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(lvl)
        console_handler.setFormatter(LoggerClass._color_formatter)
        if display_name is None:
            display_name = name
        logger = logging.getLogger(display_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    ###############
    ### Logging ###
    ###############

    def debug(self, s):
        assert (self._logger is not None)
        self._logger.debug(s)

    def info(self, s):
        assert (self._logger is not None)
        self._logger.info(s)

    def warn(self, s):
        assert (self._logger is not None)
        self._logger.warn(s)

    def error(self, s):
        assert (self._logger is not None)
        self._logger.error(s)

    def critical(self, s):
        assert (self._logger is not None)
        self._logger.critical(s)

    ####################
    ### Data logging ###
    ####################

    def record_tabular(self, key, val):
        for k, v in self._tabular:
            assert(str(key) != k)
        self._tabular.append((str(key), str(val)))

    def dump_tabular(self, print_func=None):
        if len(self._tabular) == 0:
            return ''

        ### print
        if print_func is not None:
            log_str = tabulate(self._tabular)
            for line in log_str.split('\n'):
                print_func(line)

        ### csv
        tabular_dict = dict(self._tabular)
        keys_sorted = tuple(sorted(tabular_dict.keys()))
        mode = 'a' if os.path.exists(self._csv_path) else 'w'
        with open(self._csv_path, mode) as f:
            writer = csv.writer(f)
            if mode == 'w':
                self._tabular_keys = keys_sorted
                writer.writerow(self._tabular_keys)
            else:
                assert(keys_sorted == self._tabular_keys)
            writer.writerow([tabular_dict[k] for k in self._tabular_keys])

        self._tabular = list()

logger = LoggerClass()
