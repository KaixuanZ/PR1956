#!/usr/bin/env bash
#finetune the pre-trained model (with teikoku1924) with data of teikoku1957

TrainSet=${TrainSet:-'../../personnel-records/1954/trainset/'}

WeightPath=${WeightPath:-'weight_pr1954.h5'}

GPUNum=${GPUNum:-1}

python finetune_models.py --trainset=$TrainSet --weight_path=$WeightPath --GPU_num=$GPUNum | tee ../log/July_8th_finetune_1.txt