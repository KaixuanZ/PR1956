#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../personnel-records/1956/scans/parsed'}

PagePath=${PagePath:-'../../personnel-records/1956/seg/page_rect'}

OutputPath=${OutputPath:-'../../personnel-records/1956/seg/ROI'}

#rm $OutputPath --recursive

python3 DetectROI.py --imgdir=$ImgPath --pagedir=$PagePath --outputdir=$OutputPath 2>&1 | tee log_ROI.txt