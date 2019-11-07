#!/usr/bin/env bash

declare -a Sections=("firm" "bank" "credit_union" "official_office" "supplement")
#declare -a Sections=("supplement" "firm")

ImgPath=${ImgPath:-'../raw_data/personnel-records/1956/scans/'}

RectPath=${RectPath:-'../results/personnel-records/1956/seg/'}

RowClsPath=${RowClsPath:-'../results/personnel-records/1956/cls/CRF/'}

OCRPath=${OCRPath:-'../results/personnel-records/1956/ocr/gcv_output_opt_for_name/'}

OutputPath=${OutputPath:-'../results/personnel-records/1956/res/csv_opt_for_name/'}

read -p "Do you want to remove previous output of probability in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
DELETE="N"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    DELETE='Y'
fi


for section in "${Sections[@]}"; do

    ImgPath_section="$ImgPath$section"

    RectPath_section="$RectPath$section"
    RowClsPath_section="$RowClsPath$section"
    OCRPath_section="$OCRPath$section"
    OutputPath_section="$OutputPath$section"


    if test "$DELETE" == 'Y'
    then
        echo "removing $OutputPath_section"

        rm $OutputPath_section --recursive
    fi

    mkdir $OutputPath_section

    python3 CombineAllResults.py --img_dir=$ImgPath_section \
                                    --rect_dir=$RectPath_section --row_cls_dir=$RowClsPath_section \
                                    --OCR_dir=$OCRPath_section --output_dir=$OutputPath_section

done