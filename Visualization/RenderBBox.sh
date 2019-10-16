#!/usr/bin/env bash

ImgDir=${ImgDir:-'../../raw_data/personnel-records/1956/scans/supplement'}

JsonDir=${JsonDir:-'../../results/personnel-records/1956/seg/supplement/row_rect'}

OutputDir=${OutputDir:-'../../results/personnel-records/1956/visualization'}

read -p "Do you want to remove previous output in $OutputDir? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputDir"

    rm -rf $OutputDir
fi

python3 RenderBBox.py --imgdir=$ImgDir --jsondir=$JsonDir --outputdir=$OutputDir



ImgDir='../../raw_data/personnel-records/1956/scans/firm'

JsonDir='../../results/personnel-records/1956/seg/firm/row_rect'

python3 RenderBBox.py --imgdir=$ImgDir --jsondir=$JsonDir --outputdir=$OutputDir