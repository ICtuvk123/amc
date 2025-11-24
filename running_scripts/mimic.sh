gpu_id=4
run_file=train/train_mimic.py

export ACTIVE_ROOT="/mnt/shared_workspace/pengjie/imbalance_modality"

seeds="123 132 213 231 321"
fusion_types="max mean learnable"
# seeds="123 132 213"


# args="
#     --dataset_path data/mimic4 
#     --model_config config/mimic_model_config.yml 
#     --lr 1e-4
#     --epochs 50
#     --batch_size 128
#     --modality_latent_len 48
#     --cpt_name mimic
# "

# CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${args} --seed 123


# args="
#     --dataset_path data/mimic4 
#     --model_config config/mimic_model_config.yml 
#     --lr 1e-4
#     --epochs 50
#     --batch_size 128
#     --modality_latent_len 48
#     --cpt_name mimic
# "
# for sd in $seeds
#     do
#         CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${args} --seed $sd
#     done
# python train/summary_results.py ${args}

# for ft in $fusion_types
# do
#     args="
#         --dataset_path data/mimic4 
#         --model_config config/mimic_model_config.yml 
#         --lr 1e-4
#         --epochs 50
#         --batch_size 32
#         --modality_latent_len 48
#         --cpt_name mimic
#         --fusion_type $ft
#     "
#     for sd in $seeds
#         do
#             CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${args} --seed $sd
#         done
#     python train/summary_results.py ${args}
# done

lrs="1e-4 1e-3"
batch_sizes="32 64 128"
modality_latent_lens="32 64 128"

for lr in $lrs
do
    for bs in $batch_sizes
    do
        for mll in $modality_latent_lens
        do
            args="
                --dataset_path data/mimic4 
                --model_config config/mimic_model_config.yml 
                --lr $lr
                --epochs 50
                --batch_size $bs
                --modality_latent_len $mll
                --cpt_name mimic
            "
            for sd in $seeds
            do
                CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${args} --seed $sd
            done
            python train/summary_results.py ${args}
        done
    done
done

# # CUDA_VISIBLE_DEVICES=$gpu_id python $run_file --modality screenImg --train False
# for sd in $seeds
# do

#     CUDA_VISIBLE_DEVICES=$gpu_id python $run_file --dataset_path data/mimic4 --model_config config/mimic_model_config.yml --lr 1e-5 --epochs 100 --batch_size 256 --seed $sd
# done

# python train/summary_results.py --dataset_path data/mimic4 --model_config config/mimic_model_config.yml --lr 1e-5 --epochs 100 --batch_size 256