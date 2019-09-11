#!/usr/bin/env bash

ImgDir=${ImgDir:-'../../personnel-records/1956/scans/parsed'}

JsonDir=${JsonDir:-'../../personnel-records/1956/seg/col_rect'}

OutputDir=${OutputDir:-'../../personnel-records/1956/visualization'}

read -p "Do you want to remove previous output in $OutputDir? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputDir"

    rm -rf $OutputDir
fi

python3 RenderBBox.py --imgdir=$ImgDir --jsondir=$JsonDir --outputdir=$OutputDir