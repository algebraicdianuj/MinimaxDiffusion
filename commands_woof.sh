#!/bin/bash


rm -rf logs
python sample2.py --model DiT-XL/2 --image-size 256 --ckpt logs/run-0/000-DiT-XL-2-minimax/checkpoints/0017000.pt --save-dir woof_ipc_1 --spec woof --num-samples 1 && python test_woof.py --dir woof_ipc_1 && python sample2.py --model DiT-XL/2 --image-size 256 --ckpt logs/run-0/000-DiT-XL-2-minimax/checkpoints/0017000.pt --save-dir woof_ipc_10 --spec woof --num-samples 10 && python test_woof.py --dir woof_ipc_10 && python sample2.py --model DiT-XL/2 --image-size 256 --ckpt logs/run-0/000-DiT-XL-2-minimax/checkpoints/0017000.pt --save-dir woof_ipc_50 --spec woof --num-samples 50 && python test_woof.py --dir woof_ipc_50python && generate_csv.py


