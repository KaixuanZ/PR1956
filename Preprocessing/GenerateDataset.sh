#!/usr/bin/env bash

imgpath='../../personnel-records/1954/seg/row3/'

outputpath='../../personnel-records/1954/trainset'

rm $outputpath --recursive

python3 DatasetGenerator.py --imgpath=$imgpath --labelfile='../../personnel-records/1954/csv/trainset_pr1954.csv' --labeldict='../../personnel-records/1954/csv/label_pr1954.csv' --outputpath=$outputpath

outputpath='../../personnel-records/1954/testset'

rm $outputpath --recursive

python3 DatasetGenerator.py --imgpath=$imgpath --labelfile='../../personnel-records/1954/csv/testset_pr1954.csv' --labeldict='../../personnel-records/1954/csv/label_pr1954.csv' --outputpath=$outputpath

