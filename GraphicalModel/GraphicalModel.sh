#!/usr/bin/env bash

InputPath=${InputPath:-'../../personnel-records/1954/prob/'}

OutputPath=${OutputPath:-'../../personnel-records/1954/cls/'}

LabelFile=${LabelFile:-'../../personnel-records/1954/csv/trainset_pr1954.csv'}

rm -rf $OutputPath

mkdir $OutputPath

python GraphicalModel.py --inputpath=$InputPath --outputpath=$OutputPath --labelfile=$LabelFile



# for teikoku 1957
# 98.15% without graphicalmodel
# 98.36% with manual setup
# 98.56% with automatic estimation