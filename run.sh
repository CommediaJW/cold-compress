# python generate.py \
#   --cache_strategy hybrid \
#   --cache_config fastgen \
#   --prompt long_prompt_long_output.txt \
#   --checkpoint_path /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct/model.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_pred.py \
    --model_name Meta-Llama-3.1-8B-Instruct \
    --model_name_or_path  /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --model_maxlen 4000 \
    --dataset_path /nfs/shared_LLM_dataset/LongBench \
    --dataset_name longbench \
    --output_dir ./preds/pred_longbench_Meta-Llama-3.1-8B-Instruct_0-to-32k_full \
    --write_in_time --mp_num 1 --e --min_seq_len 0