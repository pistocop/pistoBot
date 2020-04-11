#!/usr/bin/env bash

timestamp=$(date +%Y%m%d-%H%M%S)
file_name="models_$timestamp.zip"
echo "Models zip name: $file_name"

mkdir tmp
zip -r tmp/${file_name} ../data/models_trained

python ./utils/colab_file_download.py --file_path ./tmp/${file_name}

rm -r tmp