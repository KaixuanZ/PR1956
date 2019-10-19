#!/usr/bin/env bash

#declare -a Sections=("bank" "credit_union" "firm" "official_office" "supplement")
declare -a Sections=("firm" "supplement")

ProbPath=${ProbPath:-'/home/ubuntu/results/personnel-records/1956/prob/'}

LabelPath=${LabelPath:-'/home/ubuntu/results/personnel-records/1956/labeled_data/'}

OutputPath=${OutputPath:-'/home/ubuntu/results/personnel-records/1956/cls/CRF/'}

mkdir $OutputPath

read -p "Do you want to remove previous output of probability in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
DELETE="N"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    DELETE='Y'
fi

for section in "${Sections[@]}"; do

    ProbPath_section="$ProbPath$section"
    OutputPath_section="$OutputPath$section"

    if test "$DELETE" == 'Y'
    then
        echo "removing $OutputPath_section"

        rm $OutputPath_section --recursive
    fi

    mkdir $OutputPath_section

    python CRF.py --probpath=$ProbPath_section --outputpath=$OutputPath_section --labelpath=$LabelPath
done