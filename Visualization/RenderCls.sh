#!/usr/bin/env bash

ImgDir=${ImgDir:-'../../raw_data/personnel-records/1956/scans/firm'}

ROIRectDir=${ROIRectDir:-'../../results/personnel-records/1956/seg/firm/ROI_rect'}

RowRectDir=${RowRectDir:-'../../results/personnel-records/1956/seg/firm/row_rect'}

ClsDir=${ClsDir:-'../../results/personnel-records/1956/cls/CRF/firm/'}

OutputDir=${OutputDir:-'../../results/personnel-records/1956/visualization'}

read -p "Do you want to remove previous output in $OutputDir? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputDir"

    rm -rf $OutputDir
fi

python3 RenderCls.py --imgdir=$ImgDir --ROIrectdir=$ROIRectDir --rowrectdir=$RowRectDir --clsdir=$ClsDir --outputdir=$OutputDir