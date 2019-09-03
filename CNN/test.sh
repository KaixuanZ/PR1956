#!/usr/bin/env bash

TestSet=${TestSet:-'../../personnel-records/1956/testset/'}

WeightPath=${WeightPath:-'../../personnel-records/1956/models/weights39.h5'}

GPUNum=${GPUNum:-0}

python test_models.py --testset=$TestSet --weightpath=$WeightPath --GPU_num=$GPUNum