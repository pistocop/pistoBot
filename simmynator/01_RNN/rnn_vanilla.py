import argparse
import sys

import tensorflow as tf

import numpy as np
import os
import time

from simmynator.utils import print_fn


def run():
    print_fn("--------------")
    print_fn("Vanilla RNN")
    print_fn("--------------")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_input', type=str, required=False, default="../../data/inputs/raw/bible_ita.txt")
    params = parser.parse_args(argv)
    run()


if __name__ == '__main__':
    main(sys.argv[1:])
