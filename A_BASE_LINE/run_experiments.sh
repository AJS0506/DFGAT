#!/bin/bash

# 전체 데이터셋 & 시드 순회 - 병렬 실행
# 각 데이터셋(0~4)에 대해 모든 시드(1004~1008) 실행
for dataset in 0 1 2 3 4; do
    for seed in 1004 1005 1006 1007 1008; do
        echo "=============== 데이터셋 $dataset, 시드 $seed 실행 중 ==============="
        python3 baseline_train.py --gpu 0 --dataset $dataset --seed $seed --model GCN &
        python3 baseline_train.py --gpu 1 --dataset $dataset --seed $seed --model GAT &
        python3 baseline_train.py --gpu 2 --dataset $dataset --seed $seed --model SAGE &
        python3 baseline_train.py --gpu 3 --dataset $dataset --seed $seed --model GINC &
        wait  # 4개 모델 병렬 실행 완료 대기
        echo "=============== 데이터셋 $dataset, 시드 $seed 완료 ==============="
    done
    echo "=============== 데이터셋 $dataset 전체 완료 ==============="
done