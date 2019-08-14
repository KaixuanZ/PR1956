#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../personnel-records/1956/scans/parsed'}

ColPath=${ROIPath:-'../../personnel-records/1956/seg/col_rect'}

OutputPath=${OutputPath:-'../../personnel-records/1956/tmp'}

echo "removing $OutputPath"

rm $OutputPath --recursive

python3 Col2Row.py --imgdir=$ImgPath --coldir=$ColPath --outputdir=$OutputPath # 2>&1 | tee log_ROI.txt