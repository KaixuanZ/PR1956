#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../personnel-records/1956/scans/parsed/pr1956_f138_0.tif'}

RectDir=${JsonDir:-'../../personnel-records/1956/seg/col_rect/pr1956_f0138_0_1/'}

GCVDir=${GCVDir:-'../../personnel-records/1956/ocr/gcv_output/pr1956_f0138_0_1/'}

OutputDir=${OutputDir:-'../../personnel-records/1956/visualization'}

read -p "Do you want to remove previous output in $OutputDir? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputDir"

    rm -rf $OutputDir

    mkdir $OutputDir
fi

python3 RenderOCR.py --img_path=$ImgPath --rect_dir=$RectDir --gcv_dir=$GCVDir --output_dir=$OutputDir