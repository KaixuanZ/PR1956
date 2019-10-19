#!/usr/bin/env bash

#declare -a Sections=("firm" "bank" "credit_union" "official_office" "supplement")
declare -a Sections=("firm")

ImgPath=${ImgPath:-'../raw_data/personnel-records/1956/scans/'}

ColRectPath=${ColRectPath:-'../results/personnel-records/1956/seg/'}

RowRectPath=${RowRectPath:-'../results/personnel-records/1956/seg/'}

RowClsPath=${RowClsPath:-'../results/personnel-records/1956/cls/CRF/'}

OCRPath=${OCRPath:-'../results/personnel-records/1956/ocr/gcv_output/'}

OutputPath=${OutputPath:-'../results/personnel-records/1956/res/csv/'}

read -p "Do you want to remove previous output of probability in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
DELETE="N"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    DELETE='Y'
fi


for section in "${Sections[@]}"; do

    ImgPath_section="$ImgPath$section"

    ColRectPath_section="$ColRectPath$section/col_rect"
    RowRectPath_section="$RowRectPath$section/row_rect"
    RowClsPath_section="$RowClsPath$section"
    OCRPath_section="$OCRPath$section"
    OutputPath_section="$OutputPath$section"


    if test "$DELETE" == 'Y'
    then
        echo "removing $OutputPath_section"

        rm $OutputPath_section --recursive
    fi

    mkdir $OutputPath_section

    python3 CombineAllResults.py --img_dir=$ImgPath_section --col_rect_dir=$ColRectPath_section \
                                    --row_rect_dir=$RowRectPath_section --row_cls_dir=$RowClsPath_section \
                                    --OCR_dir=$OCRPath_section --output_dir=$OutputPath_section

done