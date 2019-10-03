#!/usr/bin/env bash

TestSet=${TestSet:-'../../results/personnel-records/1954/testset/'}

WeightPath=${WeightPath:-'../../results/personnel-records/1954/models/weights30.h5'}

GPUNum=${GPUNum:-0}

python test_models.py --testset=$TestSet --weightpath=$WeightPath --GPU_num=$GPUNum