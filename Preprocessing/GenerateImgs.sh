#!/usr/bin/env bash

declare -a Sections=("bank" "credit_union" "firm" "official_office" "supplement")

ImgPath=${ImgPath:-'../../raw_data/personnel-records/1954/scans/'}

RectPath=${RectPath:-'../../results/personnel-records/1954/seg/'}

OutputPath=${OutputPath:-'../../results/personnel-records/1954/seg/'}

type=${type:-'row'}

read -p "Do you want to remove previous output of $type images in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
DELETE="N"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    DELETE='Y'
fi


for section in "${Sections[@]}"; do
    echo "Processing $ImgPath$section"

    ImgPath_section="$ImgPath$section"
    OutputPath_section="$OutputPath$section/${type}_img"
    RectPath_section="$RectPath$section/${type}_rect"

    if test "$DELETE" == 'Y'
    then
        echo "removing $OutputPath_section"

        rm $OutputPath_section --recursive
    fi

    #python3 GenerateImgs.py --imgdir=$ImgPath_section --rectdir=$RectPath_section --outputdir=$OutputPath_section #2>&1 | tee log_Row.txt
done

