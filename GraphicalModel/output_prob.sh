#!/usr/bin/env bash

GPUNum=${GPUNum:-0}

InputPath=${InputPath:-'../../personnel-records/1956/seg/row_img'}

OutputPath=${OutputPath:-'../../personnel-records/1956/prob'}

read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive
fi

mkdir $OutputPath

WeightFile=${WeightFile:-'../../personnel-records/1956/pr1956.h5'}

python output_prob.py --inputpath=$InputPath --outputpath=$OutputPath --weightfile=$WeightFile --GPU_num=$GPUNum