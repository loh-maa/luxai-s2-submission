import datetime
import logging
from logging import config
import os
import pathlib
import time


def setup_logging(dname, fname='', level_console='INFO', level_file='DEBUG', logdir='.'):

    assert isinstance(dname, str) and isinstance(fname, str)
    assert '..' not in dname + fname
    assert logdir

    # The calling file may come from the __file__ builtin, extract the stem
    if fname.endswith('.py'):
        fname = pathlib.Path(fname).stem

    # datestr = datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    datestr = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    if fname:
        log_fpath = pathlib.Path(f'{logdir}/logs/{dname}/{fname}-{datestr}-{os.getpid()}.log')
    else:
        log_fpath = pathlib.Path(f'{logdir}/logs/{dname}/{datestr}-{os.getpid()}.log')
    log_fpath.parent.mkdir(parents=False, exist_ok=True)

    # Setup using dictconfig:
    cfg = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'console': {
                'format': '[%(module)16s] %(levelname)s: %(message)s',
                'datefmt': '%Y-%m-%d_%H:%M:%S'
            },
            'file': {
                # 'format': '%(asctime)s [%(module)s.%(funcName)s] [%(processName)s-%(threadName)s] %(levelname)s: %(message)s',
                'format': '%(asctime)s [%(module)s.%(funcName)s] %(levelname)s: %(message)s',
                'datefmt': '%Y-%m-%d_%H:%M:%S'
            },
            'standard': {
                'format': '%(asctime)s [%(name)s] [%(module)s] [%(levelname)s]: %(message)s'
            },
        },
        'handlers': {
            # Our default output to the console
            'console': {
                'level': level_console,
                'formatter': 'console',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',  # Default is stderr
            },
            # Primary file output, configurable for a calling module
            'rotating_file': {
                'level': level_file,
                'formatter': 'file',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_fpath,
                'mode': 'a',
                'maxBytes': 20 * 1024 * 1024,
                'backupCount': 3
            },
        },
        'loggers': {
            '': {  # root logger, we can suppress it to get rid of 3rd party libraries logging
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': False
            },
            APP_NAME: {
                'handlers': ['console', 'rotating_file'],
                'level': 'DEBUG',
                'propagate': False
            },
        },
    }

    logging.config.dictConfig(cfg)

APP_NAME = 'luxai'
log = logging.getLogger(APP_NAME)
log.SHOW_PLOTS = False

# Allow setting up logging via the logger object
log.setup_logging = setup_logging

