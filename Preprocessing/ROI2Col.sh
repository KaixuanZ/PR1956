#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../raw_data/personnel-records/1954/scans/official_office'}

ROIPath=${ROIPath:-'../../results/personnel-records/1954/seg/official_office/ROI_rect'}

OutputPath=${OutputPath:-'../../results/personnel-records/1954/seg/official_office/col_rect'}

read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive
fi

python3 ROI2Col.py --imgdir=$ImgPath --ROIdir=$ROIPath --outputdir=$OutputPath #2>&1 | tee log_ROI.txt