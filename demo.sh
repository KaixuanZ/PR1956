#!/usr/bin/env bash

#create directories for saving data
DataDir='raw_data/'
mkdir $DataDir

ResDir='results/'
mkdir $ResDir

#download one orginal image from AWS S3
AWS_S3_Path='s3://harvardaha-raw-data/personnel-records/1956/scans/firm/pr1956_f0047_0.png'


aws2 s3 cp $AWS_S3_Path $DataDir


#Start Preprocessing (results are also available on AWS S3)