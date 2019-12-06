#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../raw_data/personnel-records/1954/scans/index'}

ROIPath=${ROIPath:-'../../results/personnel-records/1954/seg/index/ROI_rect'}

OutputPath=${OutputPath:-'../../results/personnel-records/1954/seg/index/col_rect'}

read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive
fi
mkdir $OutputPath

python3 ROI2Col.py --imgdir=$ImgPath --ROIdir=$ROIPath --outputdir=$OutputPath #2>&1 | tee log_ROI.txt