#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../personnel-records/1956/scans/parsed'}

ColPath=${ROIPath:-'../../personnel-records/1956/seg/col_rect'}

OutputPath=${OutputPath:-'../../personnel-records/1956/seg/row_rect'}

read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive
fi

python3 Col2Row.py --imgdir=$ImgPath --coldir=$ColPath --outputdir=$OutputPath 2>&1 | tee log_Row.txt