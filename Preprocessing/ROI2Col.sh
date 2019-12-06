#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../personnel-records/1954/scans/index'}

ROIPath=${ROIPath:-'../../personnel-records/1954/seg/ROI'}

OutputPath=${OutputPath:-'../../personnel-records/1954/seg/col_rect'}

read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive
fi

python3 ROI2Col.py --imgdir=$ImgPath --ROIdir=$ROIPath --outputdir=$OutputPath #2>&1 | tee log_ROI.txt