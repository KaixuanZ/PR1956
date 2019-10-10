#!/usr/bin/env bash
#finetune the pre-trained model (with teikoku1924) with data of teikoku1957

TrainSet=${TrainSet:-'../../results/personnel-records/1954/trainset/'}

WeightPath=${WeightPath:-'../../results/personnel-records/1954/models/pr1954.h5'}

OutputPath=${OutputPath:-'../../results/personnel-records/1954/'}

GPUNum=${GPUNum:-0}

python finetune_models.py --trainset=$TrainSet --weight_path=$WeightPath --output_path=$OutputPath --GPU_num=$GPUNum #| tee ../log/July_8th_finetune_1.txt