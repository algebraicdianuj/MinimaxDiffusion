#!/bin/bash


rm -rf logs
torchrun --nnode=1 --master_port=25678 train_dit.py --model DiT-XL/2 --data-path cifar10_dataset --ckpt pretrained_models/DiT-XL-2-256x256.pt --global-batch-size 8 --tag minimax --ckpt-every 500 --log-every 100 --epochs 8 --condense --finetune-ipc -1 --results-dir logs/run-0 --spec cifar10
python sample.py --model DiT-XL/2 --image-size 256 --save-dir results_ipc_1 --spec cifar10 --num-samples 1
python test.py --dir results_ipc_1

python sample.py --model DiT-XL/2 --image-size 256 --save-dir results_ipc_10 --spec cifar10 --num-samples 10
python test.py --dir results_ipc_10

python sample.py --model DiT-XL/2 --image-size 256 --save-dir results_ipc_50 --spec cifar10 --num-samples 50
python test.py --dir results_ipc_50

python generate_cv.py


