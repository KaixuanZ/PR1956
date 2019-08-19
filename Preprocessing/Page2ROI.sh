#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../personnel-records/1956/scans/parsed'}

PagePath=${PagePath:-'../../personnel-records/1956/seg/page_rect'}

OutputPath=${OutputPath:-'../../personnel-records/1956/seg/ROI'}

read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive
fi

python3 Page2ROI.py --imgdir=$ImgPath --pagedir=$PagePath --outputdir=$OutputPath 2>&1 | tee log_ROI.txt