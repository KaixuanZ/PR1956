#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../personnel-records/1956/scans/parsed'}

RectPath=${RectPath:-'../../personnel-records/1956/seg/col_rect'}

OutputPath=${OutputPath:-'../../personnel-records/1956/seg/col_img'}

read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive
fi

python3 GenerateImgs.py --imgdir=$ImgPath --rectdir=$RectPath --outputdir=$OutputPath #2>&1 | tee log_Row.txt