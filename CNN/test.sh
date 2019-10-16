#!/usr/bin/env bash

TestSet=${TestSet:-'../../results/personnel-records/1956/testset/'}

WeightPath=${WeightPath:-'../../results/personnel-records/1956/models/model_pr1956.h5'}    #26-28

GPUNum=${GPUNum:-0}

python test_models.py --testset=$TestSet --weightpath=$WeightPath --GPU_num=$GPUNum