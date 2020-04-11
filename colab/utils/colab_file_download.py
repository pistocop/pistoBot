import sys
import logging
import argparse
from os.path import basename, normpath
from google.colab import files

def run(file_path: str):
    files.download("/test/")
    logging.info("Download done")


def main(argv):
    parser = argparse.ArgumentParser(prog=argv[0])
    parser.add_argument("--file_path", help="Path to file to download", type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args(argv[1:])
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    process_name = basename(normpath(argv[0]))
    logging.basicConfig(format=f"[{process_name}][%(levelname)s]: %(message)s", level=loglevel, stream=sys.stdout)
    delattr(args, "verbose")
    run(**vars(args))


if __name__ == '__main__':
    main(sys.argv)
