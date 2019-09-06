#!/usr/bin/env bash

InputPath=${InputPath:-'../../personnel-records/1956/prob/'}

OutputPath=${OutputPath:-'../../personnel-records/1956/cls/'}

Trainset=${Trainset:-'../../personnel-records/1956/csv/trainset_pr1956.csv'}

Id2Name_cls=${Id2Name_cls:-'../../personnel-records/1956/IdNameMap.json'}

Id2Name_label=${Id2Name_label:-'../../personnel-records/1956/csv/Id2Name_label.csv'}


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
# 98.15% without graphicalmodel
# 98.36% with manual setup
# 98.56% with automatic estimation