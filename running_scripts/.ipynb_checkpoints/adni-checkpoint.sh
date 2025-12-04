
###
 # @Author: PengJie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-11-20 11:31:30
 # @LastEditors: PengJie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2025-09-18 22:20:50
 # @FilePath: /imbalance_modality/running_scripts/mimic3.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
run_file=train/train_adni.py
gpu_id=6
export ACTIVE_ROOT="/mnt/shared_workspace/pengjie/imbalance_modality/"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
seeds="123 132 213 231 321"
fusion_types="max mean learnable"
# seeds="123 132 213"


lrs="1e-4 1e-5"
batch_sizes="1"
modality_latent_lens="16 32"

for lr in $lrs
do
    for bs in $batch_sizes
    do
        for mll in $modality_latent_lens
        do
            args="
                --dataset_path data/ADNI
                --model_config config/adni_model_config_diffattn.yml
                --lr $lr
                --epochs 20
                --batch_size $bs
                --modality_latent_len $mll
                --cpt_name adni_2
                --train True
            "
            for sd in $seeds
            do
                CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${args} --seed $sd
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