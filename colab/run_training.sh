#!/usr/bin/env bash

set -e
cd ..

echo "Installing common requirements..."
pip install -r requirements.txt -q

if [[ $1 == "vanilla" ]]; then
    echo "[Vanilla model choose]"

    echo "Installing requirements..."
    pip install -r ./pistoBot/01_RNN/requirements.txt

    echo "Training model..."
    python ./pistoBot/01_RNN/rnn_vanilla.py

elif [[ $1 == "gpt2-simple" ]]; then
    echo "[gpt2-simple model choose]"

    echo "Installing requirements..."
    pip install -r ./pistoBot/02_gpt2_simple/requirements.txt

    echo "Training model..."
    python ./pistoBot/02_gpt2_simple/gpt2_simple.py
else
    echo "$1 model not recognized"
fi




