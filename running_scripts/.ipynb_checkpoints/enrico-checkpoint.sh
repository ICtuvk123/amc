#!/usr/bin/env bash
set -e
gpu_id=3

# 切到项目根目录
cd "$(dirname "$0")/.."

# 适配你的机器
export ACTIVE_ROOT="$PWD"
export PYTHONPATH="$PWD"

CUDA_VISIBLE_DEVICES=$gpu_id python -m train.train \
  --epochs 100 --batch_size 128 --lr 5e-3 \
  --path model_ckpt --cpt_name enrico_img_wire \
  --enrico_path /root/amc/data/enrico \
  --model_config config/enrico_config.yml \
  --modality all
