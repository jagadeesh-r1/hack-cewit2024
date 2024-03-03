#!/bin/bash

cd /home/ubuntu/hack-cewit2024

git stash

source ./venv/bin/activate

# pip3 install --upgrade pip

pip3 install -r requirements.txt

python3 serve.py
