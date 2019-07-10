#!/usr/bin/env bash

TestSet=${TestSet:-'../../personnel-records/1954/testset/'}

WeightPath=${WeightPath:-'weight_finetune_pr1954.h5'}

GPUNum=${GPUNum:-0}

python test_models.py --testset=$TestSet --weightpath=$WeightPath --GPU_num=$GPUNum