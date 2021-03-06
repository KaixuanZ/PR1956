#!/usr/bin/env bash

#create directories for saving data
ImgDir='raw_data/'
mkdir $ImgDir

ResDir='results/'
mkdir $ResDir

#download all orginal images from AWS S3
AWS_S3_Path='s3://harvardaha-raw-data/personnel-records/1954/'

aws2 s3 sync $AWS_S3_Path $ImgDir

#download all segmentation results from AWS S3
AWS_S3_Path='s3://harvardaha-results/personnel-records/1954/'
aws2 s3 sync $AWS_S3_Path $ResDir

