import os
import sys
import yaml
import logging
import argparse
from os import makedirs
from os.path import basename, normpath, join
from datetime import datetime

from aitextgen import aitextgen
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import build_gpt2_config

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
                    min_frequency=params_ml['tokens_min_frequency'],
                    save_path=model_dir)
    logging.info("Training tokenizer completed!")

    # Train GPT-2 model
    logging.info("Training model...")
    gpt2_config = build_gpt2_config(vocab_size=params_ml['vocab_size'],
                                    max_length=params_ml['model_max_length'],
                                    dropout=params_ml['model_dropout'],
                                    n_embd=params_ml['model_n_embd'],
                                    n_layer=params_ml['model_n_layer'],
                                    n_head=params_ml['model_n_head'])
    logging.debug(f'Gpt2 configuration:{gpt2_config}')
    gpt2_model = aitextgen(config=gpt2_config,
                           vocab_file=join(model_dir, "aitextgen-vocab.json"),
                           merges_file=join(model_dir, "aitextgen-merges.txt"),
                           to_gpu=True)

    gpt2_model.train(params_data['file_path'],
                     line_by_line=False,
                     output_dir=model_dir,
                     num_steps=params_ml['train_steps'],
                     generate_every=params_ml['train_generate_every'],
                     save_every=params_ml['train_save_every'],
                     save_gdrive=False,
                     learning_rate=params_ml['train_learning_rate'],
                     batch_size=params_ml['train_batch_size'])
    logging.info("Training completed!")

    # Generate
    logging.info("Generation starting...")
    generation_folder = join(model_dir, "generation")
    os.makedirs(generation_folder, exist_ok=True)
    generation_file_path = join(generation_folder, f"{timestamp}.txt")

    gpt2_model.generate_to_file(n=params_gen['n_text'],
                                batch_size=params_gen['batch_size'],
                                destination_path=generation_file_path,
                                seed=params_gen['seed'],
                                cleanup=params_gen['cleanup'] == 'True',
                                prompt=params_gen['prefix'],
                                max_length=params_gen['max_length'],
                                temperature=params_gen['temperature'],
                                top_p=params_gen['top_p'],
                                repetition_penalty=params_gen['repetition_penalty'],
                                early_stopping=params_gen['early_stopping'],
                                num_beams=params_gen['num_beams'])
    logging.info("Generation completed!")

    # Output persist
    model_params_path = join(model_dir, 'gpt2_scratch_params.yaml')
    with open(model_params_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
    logging.debug(f"Model params saved at {model_params_path}")


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
