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
    if local:
        my_info_path = "./data/inputs/personal/my-keys.txt"
        logging.info(f"The local flag enabled. Keys at {my_info_path}")
    else:
        my_info_path = "./drive/My\ Drive/pistoBot/personal/my-keys.txt"
        logging.info(f"The colab flag enabled. Keys at {my_info_path}")
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
    my_info = None
    if path.exists(file_path):
        with open(file_path, 'r') as f:
            my_info = json.load(f)
    return my_info
