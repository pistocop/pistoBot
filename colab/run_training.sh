#!/usr/bin/env bash

set -e

cd ..

echo "Installing requirements..."
pip install -r requirements.txt -q

echo "Training model..."
python ./pistoBot/01_RNN/rnn_vanilla.py

cd ./colab/
bash ./utils/download_models_trained.sh