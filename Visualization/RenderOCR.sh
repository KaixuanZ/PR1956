#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../raw-data/personnel-records/1956/scans/firm/'}

RectDir=${JsonDir:-'../../results/personnel-records/1956/seg/firm/col_rect/'}

GCVDir=${GCVDir:-'../../results/personnel-records/1956/ocr/gcv_output/firm'}

OutputDir=${OutputDir:-'../../results/personnel-records/1956/visualization'}

read -p "Do you want to remove previous output in $OutputDir? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputDir"

    rm -rf $OutputDir

    mkdir $OutputDir
fi

python3 RenderOCR.py --img_path=$ImgPath --rect_dir=$RectDir --gcv_dir=$GCVDir --output_dir=$OutputDir