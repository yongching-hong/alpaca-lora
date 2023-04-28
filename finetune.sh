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

python finetune.py \
    --base_model 'yahma/llama-7b-hf' \
    --data_path './data/fewnerd/K200/instruct_prompt' \
    --output_dir ./lora-alpaca/fewnerd/ \
    --val_set_size 500 \
    --lora_r 16 \
    --cutoff_len 512 \
    --micro_batch_size 32 \
    --num_epochs 10 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --group_by_length \