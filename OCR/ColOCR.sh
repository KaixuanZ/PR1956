#!/usr/bin/env bash

#declare -a Sections=("bank" "credit_union" "firm" "official_office" "supplement")

declare -a Sections=("firm" "supplement")

ImgPath=${ImgPath:-'../../results/personnel-records/1956/seg/'}

OutputPath=${OutputPath:-'../../results/personnel-records/1956/ocr/gcv_output/'}

read -p "Do you want to remove previous output of $type images in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
DELETE="N"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    DELETE='Y'
fi

for section in "${Sections[@]}"; do


    ImgPath_section="$ImgPath$section/col_img/"
    OutputPath_section="$OutputPath$section"

    if test "$DELETE" == 'Y'
    then
        echo "removing $OutputPath_section"

        rm $OutputPath_section --recursive
    fi
    echo "Processing $ImgPath_section"
    python3 ColOCR.py --imgdir=$ImgPath_section --outputdir=$OutputPath_section
done