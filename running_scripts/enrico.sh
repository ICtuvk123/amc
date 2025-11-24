gpu_id=3
run_file=train/train.py 

export ACTIVE_ROOT="/mnt/shared_workspace/pengjie/active_missing"
# CUDA_VISIBLE_DEVICES=$gpu_id python $run_file --modality screenImg --train False
CUDA_VISIBLE_DEVICES=$gpu_id python $run_file --path model_ckpt/weighted_fusion