#!/usr/bin/env bash

declare -a Sections=("supplement" "firm")

InputPath=${InputPath:-'../../raw_data/personnel-records/1956/scans/'}

OutputPath=${OutputPath:-'../../results/personnel-records/1956/seg/'}

read -p "Do you want to remove previous output of probability in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
DELETE="N"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    DELETE='Y'
fi

for section in "${Sections[@]}"; do

    InputPath_section="$InputPath$section"
    ColPath_section="$OutputPath$section/col_rect"
    OutputPath_section="$OutputPath$section/row_rect"

    if test "$DELETE" == 'Y'
    then
        echo "removing $OutputPath_section"

        rm $OutputPath_section --recursive
    fi

    mkdir $OutputPath_section

    python3 Col2Row.py --imgdir=$InputPath_section --coldir=$ColPath_section --outputdir=$OutputPath_section  # 2>&1 | tee "../../results/personnel-records/1956/log/log_Col2Row_$section.txt"

done