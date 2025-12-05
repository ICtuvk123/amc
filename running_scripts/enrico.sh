#!/bin/bash
gpu_id=3
run_file=train/train.py

export ACTIVE_ROOT="/root/amc"
export PYTHONPATH="/root/amc:/root/amc/scattermoe:$PYTHONPATH"
# CUDA_VISIBLE_DEVICES=$gpu_id python $run_file --modality screenImg --train False
CUDA_VISIBLE_DEVICES=$gpu_id python $run_file --path model_ckpt/weighted_fusion --fusion_type weighted_fusion --modality_latent_len 16