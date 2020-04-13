#!/usr/bin/env bash

set -e

timestamp=$(date +%Y%m%d-%H%M%S)
file_name="models_$timestamp.zip"
tmp_folder="tmp_$timestamp"
zip_file=${tmp_folder}/${file_name}
mkdir ${tmp_folder}

echo "Models zip name: $file_name"
echo "Tmp zip folder: $tmp_folder"

echo "Compressing 'models_trained' folder..."
7z a  ${zip_file} "../data/models_trained"


echo "Downloading zip file: $zip_file"
python ./utils/colab_file_download.py --file_path ${zip_file}