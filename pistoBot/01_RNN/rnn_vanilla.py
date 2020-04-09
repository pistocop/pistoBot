"""
Code used to study the TF official code:
https://www.tensorflow.org/tutorials/text/text_generation
"""
import datetime
import os
import sys
import logging
import argparse
from typing import Dict, List, Tuple
from os.path import basename, normpath

import tensorflow as tf
import knockknock
import numpy as np
import nltk

nltk.download("punkt")


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
    target_text = chunk[1:]
    return input_text, target_text


def run():
    """
    [My notes]

    Note 1: we will predict not only one token, but a phrase of len `input_seq_length`, where
            the system hides the first token, and a new token will be predicted
    """
    # TODO move as program input
    input_file_path = "./data/inputs/raw/simmy_join.txt"
    input_file_encoding = "utf-8"
    input_token_level = "word"
    input_seq_length = 3
    input_lowercase = True
    input_batch_size = 64
    input_buffer_size = None
    # -----------

    tf.random.set_seed(42)

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
    dataset_ml = dataset_ml.shuffle(buffer_size, reshuffle_each_iteration=True)
    dataset_ml = dataset_ml.batch(input_batch_size, drop_remainder=True)

    logging.info("----------------------------")
    logging.info("[Example of ML batch]")
    logging.info("train --> label")
    for x_batch, y_batch in dataset_ml.take(1):
        # Take one batch
        for idx, (batch_el_x, batch_el_y) in enumerate(zip(x_batch, y_batch)):
            logging.info(f"{idx} | {text_decoder[batch_el_x.numpy()]} --> {text_decoder[batch_el_y.numpy()]}")
    logging.info("----------------------------")

    # Model
    # TODO move under input params
    input_embedding_dim = 256
    input_rnn_units = 1024
    input_dropout = 0.3
    input_epochs = 10
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

    # Network test (from ingestion to prediction)
    for x, y in dataset_ml.take(1):
        batch_prediction_example = model(x)
        logging.info(f"prediction shape: {batch_prediction_example.shape} | [batch, seq_len, vocab_size]")

    # take 1 element according to categorical distribution given by last NN dense layer
    # Note: It is important to sample from this distribution as taking the argmax
    # of the distribution can easily get the model stuck in a loop.
    sampled_indices = tf.random.categorical(batch_prediction_example[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    logging.info("Example Input: {}".format(repr(" ".join(text_decoder[x[0].numpy()]))))
    logging.info("Example NN output: {}".format(repr(" ".join(text_decoder[sampled_indices]))))

    # NN train
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    example_batch_loss = loss(x, batch_prediction_example)

    model.compile(optimizer='adam', loss=loss)
    checkpoint_dir = 'data/models_trained/{}'.format(datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S'))
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")  # The system will fill _epoch_
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                             save_weights_only=True)
    history = model.fit(dataset_ml, epochs=input_epochs, callbacks=[checkpoint_callback])
    logging.info("Training completed")
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
