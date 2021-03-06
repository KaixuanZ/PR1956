#!/usr/bin/env bash

InputPath=${InputPath:-'../../raw_data/personnel-records/1954/scans/bank/'}

OutputPath=${OutputPath:-'../../results/personnel-records/1954/seg/bank/page_rect/'}


read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive
fi

python3 Img2Page.py --inputdir=$InputPath --outputdir=$OutputPath   # 2>&1 | tee log_Img2Page.txt

./RemoveEmptyPage.sh