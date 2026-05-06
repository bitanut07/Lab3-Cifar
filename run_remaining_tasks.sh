#!/usr/bin/env bash
set -euo pipefail

cd "/Users/lap14568/HocThongKe/Lab3-Cifar-2"

# Nếu đã có data/cifar10.npz thì bỏ qua bước configure này.
python3 main.py configure --dataset cifar10

# Task 1 + 2
python3 main.py train --model cifar_mlp --epochs 25 --batch-size 128 --validation-split 0.1
python3 main.py test --model cifar_mlp

python3 main.py train --model cnn --epochs 20 --batch-size 128 --validation-split 0.1
python3 main.py test --model cnn

# Task 3 (augmentation)
python3 main.py train --model cnn_aug --epochs 25 --batch-size 128 --validation-split 0.1
python3 main.py test --model cnn_aug

# Task 5 (improvement: GAP + early stopping)
python3 main.py train --model cnn_gap --epochs 25 --batch-size 128 --validation-split 0.1
python3 main.py test --model cnn_gap

# Tổng hợp kết quả
python3 main.py summary

echo "Done. Check results/ for json, md, curves, confusion."
