#!/usr/bin/env bash

declare -a Sections=("firm" "supplement")

InputPath=${InputPath:-'../../results/personnel-records/1954/seg/'}

ClsPath=${ClsPath:-'../../results/personnel-records/1954/cls/CRF/'}

read -p "Do you want to remove previous output in $InputPath/section/col_img_opt_for_name? (y/n) " -n 1 -r
echo -e "\n"
DELETE="N"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    DELETE='Y'
fi

for section in "${Sections[@]}"; do

    ColImgPath_section="$InputPath$section/col_img"
    ClsPath_section="$ClsPath$section"
    ColRectPath_section="$InputPath$section/col_rect"
    RowRectPath_section="$InputPath$section/row_rect"
    OutputPath_section="$InputPath$section/col_img_opt_for_name"

    if test "$DELETE" == 'Y'
    then
        echo "removing $OutputPath_section"

        rm $OutputPath_section --recursive
    fi

    mkdir $OutputPath_section

    python3 MovingNames.py --imgdir=$ColImgPath_section --clsdir=$ClsPath_section --colrectdir=$ColRectPath_section --rowrectdir=$RowRectPath_section --outputdir=$OutputPath_section  # 2>&1 | tee "../../results/personnel-records/1956/log/log_Col2Row_$section.txt"

done