import sys
import logging
import argparse
from itertools import dropwhile

import tensorflow as tf
import knockknock
import humanfriendly
import numpy as np
import nltk
from typing import Dict, List, Tuple
from os.path import basename, normpath


def read_dataset(file_path: str, file_encoding: str = "utf-8") -> str:
    text = open(file_path, 'r', encoding=file_encoding).read()
    return text


def _custom_tokenizer(text: str, level: str) -> str:
    # TODO nltk don't preserve \n char
    if level == "word":
        text = nltk.word_tokenize(text, language='italian', preserve_line=False)
    return text


def create_vocabulary(text: str) -> Tuple[dict, np.ndarray]:
    text_tokens = sorted(set(text))  # text_chars = List
    token2idx = {u: i for i, u in enumerate(text_tokens)}
    idx2token = np.array(text_tokens)
    return token2idx, idx2token


def input_encoder(text: str, encoder: Dict[str, int]) -> np.ndarray:
    text_encoded = np.array([encoder[token] for token in text])
    return text_encoded


def _custom_parser(text: str, lowercase: bool, stop_words: List[str] = None) -> str:
    # TODO enhance the parser (auto_nlp?)
    text = text.lower()
    return text


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[-1]
    return input_text, target_text


def run():
    # TODO move as program input
    input_file_path = "./data/inputs/raw/simmy_join.txt"
    input_file_encoding = "utf-8"
    input_token_level = "word"
    input_seq_length = 2
    input_lowercase = True
    input_batch_size = 64
    input_buffer_size = None
    # -----------

    # Dataset
    dataset_text = read_dataset(input_file_path, input_file_encoding)
    dataset_text = _custom_parser(dataset_text, lowercase=input_lowercase)
    dataset_text = _custom_tokenizer(dataset_text, input_token_level)
    text_encoder, text_decoder = create_vocabulary(dataset_text)
    vocab_size = len(text_encoder)
    logging.info(f"[{basename(input_file_path)} is composed by {len(dataset_text)}|{vocab_size} tot|unique tokens]")
    dataset_encoded = input_encoder(dataset_text, text_encoder)
    dataset_ml = tf.data.Dataset.from_tensor_slices(dataset_encoded)
    dataset_ml = dataset_ml.batch(batch_size=input_seq_length + 1, drop_remainder=True)  # +1 is the label
    dataset_ml = dataset_ml.map(split_input_target)
    buffer_size = input_buffer_size if input_buffer_size else len(dataset_encoded)
    dataset_ml.shuffle(buffer_size, seed=42, reshuffle_each_iteration=True)
    dataset_batches = dataset_ml.batch(input_batch_size, drop_remainder=True)

    logging.info("----------------------------")
    logging.info("[Example of ML batch]")
    logging.info("train --> label")
    for x_batch, y_batch in dataset_batches.take(1):
        for idx, (x, y) in enumerate(zip(x_batch, y_batch)):
            logging.info(f"{idx} | {text_decoder[x.numpy()]} --> {text_decoder[y]}")
    logging.info("----------------------------")

    # Model
    # TODO move under input params
    input_embedding_dim = 256
    input_rnn_units = 1024
    input_dropout = 0.3
    # ---------------------------
    model = tf.keras.Sequential(name="my_vanilla_rnn")
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size,
                                        output_dim=input_embedding_dim,
                                        batch_input_shape=[input_batch_size, input_seq_length]))
    model.add(tf.keras.layers.GRU(units=input_rnn_units,
                                  return_sequences=True,
                                  stateful=True,
                                  recurrent_initializer='glorot_uniform',
                                  dropout=input_dropout))
    model.add(tf.keras.layers.Dense(vocab_size))
    logging.info(model.summary())
    for x, y in dataset_batches.take(1):
        p = model(x)
        logging.info(f"prediction shape: {p.shape} | [batch, seq_len, vocab_size]")
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
