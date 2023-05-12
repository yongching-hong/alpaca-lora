# RUN_NAME='t5-v11-large'
# OUTPUT_DIR=./lora-alpaca/fewnerd/${RUN_NAME}
# python finetune.py \
#     --model_type 't5' \
#     --base_model 'google/t5-v1_1-large' \
#     --data_path './data/fewnerd/uie/mix' \
#     --output_dir ${OUTPUT_DIR} \
#     --val_set_size 500 \
#     --cutoff_len 900 \
#     --micro_batch_size 4 \
#     --num_epochs 20 \
#     --lora_target_modules None \
#     --wandb_project 'instructUIE' \
#     --wandb_run_name ${RUN_NAME} \

RUN_NAME='flan-t5-xxl-mix'
OUTPUT_DIR=./lora-alpaca/fewnerd/${RUN_NAME}
python finetune.py \
    --model_type 't5' \
    --base_model 'google/flan-t5-xxl' \
    --data_path './data/fewnerd/uie/mix' \
    --output_dir ${OUTPUT_DIR} \
    --val_set_size 500 \
    --lora_r 16 \
    --cutoff_len 900 \
    --micro_batch_size 8 \
    --num_epochs 20 \
    --lora_target_modules '[q,v]' \
    --wandb_project 'test' \
    --wandb_run_name ${RUN_NAME}