# python finetune.py \
#     --base_model 'decapoda-research/llama-7b-hf' \
#     --data_path './data/alpaca_data_gpt4.json' \
#     --output_dir './lora-alpaca/gpt4/7b' \
#     --learning_rate 2e-5 \
#     --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
#     --cutoff_len 512 \
#     --lora_r 16 \
#     --batch_size 128 \
#     --micro_batch_size 64 \
#     --group_by_length \
#     --num_epochs 10 \

RUN_NAME='test-llama-13b'
OUTPUT_DIR=./lora-alpaca/fewnerd/${RUN_NAME}
python finetune.py \
    --base_model 'yahma/llama-13b-hf' \
    --data_path './data/fewnerd/uie/instruct' \
    --output_dir ${OUTPUT_DIR} \
    --val_set_size 500 \
    --lora_r 16 \
    --cutoff_len 900 \
    --micro_batch_size 4 \
    --num_epochs 10 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --group_by_length \
    --wandb_project 'test' \
    --wandb_run_name ${RUN_NAME} \