#!/usr/bin/env bash
#training model with data from teikoku1924, heigh~=800, width~=80

TrainSet=${TrainSet:-'../../trainset/teikoku1924/'}

GPUNum=${GPUNum:-0}

python train_models.py --trainset=$TrainSet --GPU_num=$GPUNum | tee ../log/June_12th_train.txt