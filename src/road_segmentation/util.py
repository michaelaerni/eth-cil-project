import logging
import os


DEFAULT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
DEFAULT_DATA_DIR = os.path.join(DEFAULT_BASE_DIR, 'data')
DEFAULT_LOG_DIR = os.path.join(DEFAULT_BASE_DIR, 'log')


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s [%(name)s]: %(message)s")
