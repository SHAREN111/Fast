#!/bin/bash

# 获取命令行参数，设置默认值
EXP_NAME="${1:-raw}"
INDEX="${2:-0}"
MODE="${3:-raw}"
SPARSE="${4:-0.8}"
# 并行执行所有命令
CUDA_VISIBLE_DEVICES=0 python3 ./evaluation/vbench_480.py --exp_name "$EXP_NAME" --index "$INDEX" --start_mode 0 --mode "$MODE" --sparse "$SPARSE" &
CUDA_VISIBLE_DEVICES=1 python3 ./evaluation/vbench_480.py --exp_name "$EXP_NAME" --index "$INDEX" --start_mode 1 --mode "$MODE" --sparse "$SPARSE" &
CUDA_VISIBLE_DEVICES=2 python3 ./evaluation/vbench_480.py --exp_name "$EXP_NAME" --index "$INDEX" --start_mode 2 --mode "$MODE" --sparse "$SPARSE" &
CUDA_VISIBLE_DEVICES=3 python3 ./evaluation/vbench_480.py --exp_name "$EXP_NAME" --index "$INDEX" --start_mode 3 --mode "$MODE" --sparse "$SPARSE" &
CUDA_VISIBLE_DEVICES=4 python3 ./evaluation/vbench_480.py --exp_name "$EXP_NAME" --index "$INDEX" --start_mode 4 --mode "$MODE" --sparse "$SPARSE" &
CUDA_VISIBLE_DEVICES=5 python3 ./evaluation/vbench_480.py --exp_name "$EXP_NAME" --index "$INDEX" --start_mode 5 --mode "$MODE" --sparse "$SPARSE" &
CUDA_VISIBLE_DEVICES=6 python3 ./evaluation/vbench_480.py --exp_name "$EXP_NAME" --index "$INDEX" --start_mode 6 --mode "$MODE" --sparse "$SPARSE" &
CUDA_VISIBLE_DEVICES=7 python3 ./evaluation/vbench_480.py --exp_name "$EXP_NAME" --index "$INDEX" --start_mode 7 --mode "$MODE" --sparse "$SPARSE" &
 
# 等待所有后台进程完成
wait