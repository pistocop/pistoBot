#!/usr/bin/env bash

timestamp=$(date +%Y%m%d-%H%M%S)
file_name="models_$timestamp.zip"

tmp_folder="tmp_$timestamp"
echo "Models zip name: $file_name"
echo "Tmp zip folder: $tmp_folder"

mkdir ${tmp_folder}
zip -r ${tmp_folder}/${file_name} ../data/models_trained
python ./utils/colab_file_download.py --file_path ./tmp/${file_name} -v