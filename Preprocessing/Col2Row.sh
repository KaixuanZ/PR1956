#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../raw_data/personnel-records/1954/scans/bank'}

ColPath=${ColPath:-'../../results/personnel-records/1954/seg/bank/col_rect'}

OutputPath=${OutputPath:-'../../results/personnel-records/1954/seg/bank/row_rect'}

read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive
fi

python3 Col2Row.py --imgdir=$ImgPath --coldir=$ColPath --outputdir=$OutputPath #2>&1 | tee log_Row.txt