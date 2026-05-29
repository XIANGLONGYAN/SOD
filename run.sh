export MODEL_PATH=/jizhicfs/jackxlyan/pretrained/qwen3/qwen-3-1.7b
export TRAIN_DATA=/jizhicfs/jackxlyan/dataset/sod_data/sft_3k/full_sft_3k_shuffled_v4.parquet
export SAVE_PATH=./checkpoint/qwen3_sft
export NPROC_PER_NODE=8

source examples/SOD/run_sft.sh
