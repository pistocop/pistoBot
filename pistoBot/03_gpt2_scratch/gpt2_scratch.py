import os
import sys
import yaml
import logging
import argparse
from os import makedirs
from os.path import basename, normpath, join
from datetime import datetime

from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen

sys.path.append("./")  # needed 4 utils imports - created according to launcher
from pistoBot.utils.general_utils import my_init, load_yaml


def run(path_params: str):
    # Input
    params = load_yaml(path_params)
    params_data = params['data']
    params_ml = params['ml']
    params_gen = params['generation']
    logging.debug(f"Params: {params}")

    # Init
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    run_name = f"03_gpt2scratch_{timestamp}"
    model_dir = join(params_ml['save_path'], run_name)
    os.makedirs(model_dir, exist_ok=True)

    # Train tokenizer
    logging.info("Training tokenizer...")
    dropout = params_ml['tokenizer_dropout'] if params_ml['tokenizer_dropout'] != 0.0 else None
    train_tokenizer(files=params_data['file_path'],
                    dropout=dropout,
                    vocab_size=params_ml['vocab_size'],
                    min_frequency=params_ml['min_frequency'],
                    save_path=model_dir)
    logging.info("Training tokenizer completed!")


def main(argv):
    parser = argparse.ArgumentParser(prog=argv[0])
    parser.add_argument("--path_params", help="Path to rnn YAML params",
                        default="./pistoBot/03_gpt2_scratch/gpt2_scratch_params.yaml")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args(argv[1:])
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    process_name = basename(normpath(argv[0]))
    logging.basicConfig(format=f"[{process_name}][%(levelname)s]: %(message)s", level=loglevel, stream=sys.stdout)
    run_initialized = my_init(run)
    delattr(args, "verbose")
    run_initialized(**vars(args))


if __name__ == '__main__':
    main(sys.argv)
