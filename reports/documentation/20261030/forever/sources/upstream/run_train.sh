#!/bin/bash

export CUDA_VISIBLE_DEVICES=5   

export WANDB_MODE=offline                    
export WANDB_PROJECT=cl_memory_vanilla       
export WANDB_API_KEY="dummy_offline_key"    

export WANDB_DIR="/workspace/CL_exp/wandblog/exp_22_Qwen3-0.6B_new72_errloss_datalong_id3456"  
mkdir -p "$WANDB_DIR"                        

begin_id=0

MEM_EPOCHS=2        
STEPS_PER_DAY=24   
MEM_RATIO=2        

get_end_id () {
python - <<PY
from utils.dataset_order import get_dataset_order
print(len(get_dataset_order(int($1))))
PY
}

for data_id in 1 2 3 4 5 6  
do
    end_id=$(get_end_id ${data_id})
    echo "dataset_id=${data_id} has ${end_id} tasks"

    for ((ORDERR=$begin_id; ORDERR<end_id; ORDERR++))
    do
        
        torchrun \
            --nproc_per_node=1 \
            --master_port=1258 \
            finetune_cl_ours_qwen.py \
            --base_model '/workspace/Qwen3-0.6B' \
            --num_epochs 10 \
            --first_task_epochs 60 \
            --group_by_length \
            --lora_target_modules '["q_proj","v_proj"]' \
            --micro_batch_size 4 \
            --batch_size 32 \
            --max_input_length 512 \
            --dataset_id ${data_id} \
            --task_id ${ORDERR} \
            --output_path "/workspace/CL_exp/checkpoints/exp_22_Qwen3-0.6B_new72_errloss_datalong_id3456/" \
            --wandb_project "${WANDB_PROJECT}" \
            --memory_epochs ${MEM_EPOCHS} \
            --steps_per_day ${STEPS_PER_DAY} \
            --memory_data_ratio ${MEM_RATIO}
    done
done
