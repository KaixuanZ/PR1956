#!/usr/bin/env bash

#declare -a Sections=("bank" "credit_union" "firm" "official_office" "supplement")
declare -a Sections=("official_office" "supplement")
GPUNum=${GPUNum:-0}

ImgPath=${ImgPath:-'../../results/personnel-records/1954/seg/'}

OutputPath=${OutputPath:-'../../results/personnel-records/1954/prob/'}

WeightFile=${WeightFile:-'../../results/personnel-records/1954/models/model_pr1954.h5'}

mkdir $OutputPath

read -p "Do you want to remove previous output of probability in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
DELETE="N"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    DELETE='Y'
fi

for section in "${Sections[@]}"; do

    ImgPath_section="$ImgPath$section/row_img"
    OutputPath_section="$OutputPath$section"

    if test "$DELETE" == 'Y'
    then
        echo "removing $OutputPath_section"

        rm $OutputPath_section --recursive
    fi

    mkdir $OutputPath_section

    python output_prob.py --inputpath=$ImgPath_section --outputpath=$OutputPath_section --weightfile=$WeightFile --GPU_num=$GPUNum
done