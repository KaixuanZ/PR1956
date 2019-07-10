#!/usr/bin/env bash

GPUNum=${3:-0}

InputPath=../../personnel-records/1954/seg/row2/

OutputPath=${OutputPath:-'../../personnel-records/1954/prob'}

rm $OutputPath --recursive

mkdir $OutputPath

WeightFile=${WeightFile:-'../CNN/weight_finetune_pr1954.h5'}

python output_prob.py --inputpath=$InputPath --outputpath=$OutputPath --weightfile=$WeightFile --GPU_num=$GPUNum
