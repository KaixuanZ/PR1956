#!/usr/bin/env bash

imgpath='../../results/personnel-records/1956/seg/firm/row_img'

outputpath='../../results/personnel-records/1956/trainset'

read -p "Do you want to remove previous output in $outputpath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $outputpath"

    rm $outputpath --recursive
fi

#python3 DatasetGenerator.py --imgpath=$imgpath --labelfile='../../results/personnel-records/1956/labeled_data/trainset_pr1956.csv' --labeldict='../../results/personnel-records/1956/labeled_data/Id2Name_label.csv' --outputpath=$outputpath

python3 DatasetGenerator.py --imgpath=$imgpath --labelfile='../../results/personnel-records/1956/labeled_data/PR1956_extra_data.csv' --labeldict='../../results/personnel-records/1956/labeled_data/Id2Name_label.csv' --outputpath=$outputpath


#outputpath='../../results/personnel-records/1954/testset'

#read -p "Do you want to remove previous output in $outputpath? (y/n) " -n 1 -r
#echo -e "\n"
#if [[ $REPLY =~ ^[Yy]$ ]]
#then
#    echo "removing $outputpath"
#
#    rm $outputpath --recursive
#fi

#python3 DatasetGenerator.py --imgpath=$imgpath --labelfile='../../results/personnel-records/1956/labeled_data/testset_pr1956.csv' --labeldict='../../results/personnel-records/1956/labeled_data/Id2Name_label.csv' --outputpath=$outputpath

