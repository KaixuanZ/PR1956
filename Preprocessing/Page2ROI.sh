#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../raw_data/personnel-records/1954/scans/official_office'}

PagePath=${PagePath:-'../../results/personnel-records/1954/seg/official_office/page_rect'}

OutputPath=${OutputPath:-'../../results/personnel-records/1954/seg/official_office/ROI_rect'}

read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive
fi

python3 Page2ROI.py --imgdir=$ImgPath --pagedir=$PagePath --outputdir=$OutputPath #2>&1 | tee log_ROI.txt


