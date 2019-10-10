#!/usr/bin/env bash

InputPath=${InputPath:-'../../results/personnel-records/1954/prob/firm/'}

OutputPath=${OutputPath:-'../../results/personnel-records/1954/cls/firm/'}

Trainset=${Trainset:-'../../results/personnel-records/1954/labeled_data/trainset_pr1954.csv'}

Id2Name_cls=${Id2Name_cls:-'../../results/personnel-records/1954/labeled_data/IdNameMap.json'}

Id2Name_label=${Id2Name_label:-'../../results/personnel-records/1954/labeled_data/Id2Name_label.csv'}


read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive

    mkdir $OutputPath
fi

python GraphicalModel.py --inputpath=$InputPath --outputpath=$OutputPath --trainset=$Trainset --id2name_cls=$Id2Name_cls --id2name_label=$Id2Name_label

python CountCName.py

# for teikoku 1957
# 98.4% without graphicalmodel
# 99.4% with manual setup
# 99.7% with automatic estimation