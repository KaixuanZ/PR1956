#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../results/personnel-records/1956/seg/firm/col_img/'}

GCVDir=${GCVDir:-'../../results/personnel-records/1956/ocr/gcv_output/firm'}

OutputDir=${OutputDir:-'../../results/personnel-records/1956/visualization_OCR'}

read -p "Do you want to remove previous output in $OutputDir? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputDir"

    rm -rf $OutputDir

    mkdir $OutputDir
fi

python3 RenderOCR.py --img_path=$ImgPath --gcv_dir=$GCVDir --output_dir=$OutputDir