#!/bin/bash

mkdir pretrained_models
cd pretrained_models
wget https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
cd ..
python create_cifardataset.py

