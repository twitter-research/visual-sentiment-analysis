"""
Copyright 2020 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""
import logging


LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
LOG_TIME_FORMAT = None


def config_basic(level=logging.INFO, fmt=LOG_FORMAT, datefmt=LOG_TIME_FORMAT):
  logging.basicConfig(format=fmt, datefmt=datefmt, level=level)


def add_file_handler(logger, log_file, fmt=LOG_FORMAT, datefmt=LOG_TIME_FORMAT, add_to_root=False):
  logFormatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
  fileHandler = logging.FileHandler(log_file)
  fileHandler.setFormatter(logFormatter)
  if add_to_root:
    logger.root.addHandler(fileHandler)
  else:
    logger.addHandler(fileHandler)
  return logger
