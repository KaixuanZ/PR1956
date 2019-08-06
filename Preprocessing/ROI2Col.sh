#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../personnel-records/1956/scans/parsed'}

ROIPath=${ROIPath:-'../../personnel-records/1956/seg/ROI'}

OutputPath=${OutputPath:-'../../personnel-records/1956/seg/col_rect'}

rm $OutputPath --recursive

python3 ROI2Col.py --imgdir=$ImgPath --ROIdir=$ROIPath --outputdir=$OutputPath 2>&1 | tee log_ROI.txt