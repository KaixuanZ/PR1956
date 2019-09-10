#!/usr/bin/env bash

ImgPath=${ImgPath:-'../personnel-records/1956/scans/parsed'}

ColRectPath=${ColRectPath:-'../personnel-records/1956/seg/col_rect'}

RowRectPath=${RowRectPath:-'../personnel-records/1956/seg/row_rect'}

RowClsPath=${RowClsPath:-'../personnel-records/1956/cls/'}

OCRPath=${OCRPath:-'../personnel-records/1956/ocr/gcv_output'}

OutputPath=${OutputPath:-'../personnel-records/1956/res/'}

read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive
fi

python3 CombineAllResults.py --img_dir=$ImgPath --col_rect_dir=$ColRectPath --row_rect_dir=$RowRectPath --row_cls_dir=$RowClsPath --OCR_dir=$OCRPath --output_dir=$OutputPath