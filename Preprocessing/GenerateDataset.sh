#!/usr/bin/env bash

imgpath='../../personnel-records/1956/seg/row_img/'

outputpath='../../personnel-records/1956/trainset'

read -p "Do you want to remove previous output in $outputpath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $outputpath"

    rm $outputpath --recursive
fi


python3 DatasetGenerator.py --imgpath=$imgpath --labelfile='../../personnel-records/1956/csv/trainset_pr1956.csv' --labeldict='../../personnel-records/1956/csv/Id2Name_label.csv' --outputpath=$outputpath


outputpath='../../personnel-records/1956/testset'

read -p "Do you want to remove previous output in $outputpath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $outputpath"

    rm $outputpath --recursive
fi

python3 DatasetGenerator.py --imgpath=$imgpath --labelfile='../../personnel-records/1956/csv/testset_pr1956.csv' --labeldict='../../personnel-records/1956/csv/Id2Name_label.csv' --outputpath=$outputpath

