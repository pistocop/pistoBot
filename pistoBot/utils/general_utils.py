import json
import logging

import yaml
from os import path
from nltk import download
import tensorflow as tf


def load_yaml(path: str):
    with open(path, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_dict


def my_init(run, local: bool = None):
    download("punkt")
    if tf.__version__ >= '2.0.0':
        tf.random.set_seed(42)

    # Enable telegram start and stop notification
    my_info_path = "./data/inputs/personal/my-keys.txt" if local else "../drive/My Drive/pistoBot/personal/my-keys.txt"
    my_info = get_my_info(my_info_path)
    if my_info:
        logging.info("Telegram notification enabled")
        from knockknock import telegram_sender
        telegram_decorator = telegram_sender(token=my_info['telegram_token'],
                                             chat_id=my_info['telegram_chat_id'])
        run = telegram_decorator(run)
    return run


def get_my_info(file_path: str) -> dict:
    """
    Read and return all personal info.
    Used to load personal keys used in useful features.

    e.g. load telegram token to send notification of start and stop training.
    """
    if path.exists(file_path):
        with open(file_path, 'r') as f:
            my_info = json.load(f)
            logging.debug(f"Keys file at {file_path} loaded with {my_info.keys()} values")
    else:
        logging.warning(f"Keys file at {file_path} not found")
        raise ValueError(f"Keys file at {file_path} not found")
    return my_info
