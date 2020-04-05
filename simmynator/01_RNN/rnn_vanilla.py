import sys
import logging
import argparse
import tensorflow as tf
import knockknock
import humanfriendly
import numpy as np
from typing import Dict, List, Tuple
from os.path import basename, normpath


def read_dataset(file_path: str, file_encoding: str = "utf-8") -> str:
    text = open(file_path, 'r', encoding=file_encoding).read()
    return text


def _custom_tokenizer(text: str, level: str) -> str:
    # TODO more performing splitter
    if level == "word":
        text = text.replace("\n", " ")
        text += " \n"
        text = text.split(sep=" ")
    return text


def create_vocabulary(text: str) -> Tuple[dict, np.ndarray]:
    text_tokens = sorted(set(text))  # text_chars = List
    token2idx = {u: i for i, u in enumerate(text_tokens)}
    idx2token = np.array(text_tokens)
    return token2idx, idx2token


def input_encoder(text: str, encoder: Dict[str, int]) -> np.ndarray:
    text_encoded = np.array([encoder[token] for token in text])
    return text_encoded


def run():
    # TODO move as program input
    input_file_path = "./data/inputs/raw/simmy_join.txt"
    input_file_encoding = "utf-8"
    input_token_level = "char"
    input_seq_length = 2
    # -----------

    # Dataset
    dataset_text = read_dataset(input_file_path, input_file_encoding)
    dataset_text = _custom_tokenizer(dataset_text, input_token_level)
    logging.info(f"[File {basename(input_file_path)} is composed by {len(dataset_text)} tokens]")
    text_encoder, text_decoder = create_vocabulary(dataset_text)
    dataset_encoded = input_encoder(dataset_text, text_encoder)
    dataset_ml = tf.data.Dataset.from_tensor_slices(dataset_encoded)
    dataset_batches = dataset_ml.batch(batch_size=input_seq_length, drop_remainder=True)
    logging.info("----------------------------")
    logging.info("[Example of ML train]")
    logging.info("train\t| label")
    for batch in dataset_batches.take(5):
        x, y = batch.numpy()
        logging.info(f"{text_decoder[x]} ({x}) --> {text_decoder[y]} ({y})")
    logging.info("----------------------------")

    # ML

    pass


def main(argv):
    parser = argparse.ArgumentParser(prog=argv[0])
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args(argv[1:])
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    process_name = basename(normpath(argv[0]))
    logging.basicConfig(format=f"[{process_name}][%(levelname)s]: %(message)s", level=loglevel, stream=sys.stdout)
    delattr(args, "verbose")
    run(**vars(args))


if __name__ == '__main__':
    main(sys.argv)
