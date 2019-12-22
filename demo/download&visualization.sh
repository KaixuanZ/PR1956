#!/usr/bin/env bash


#####download one image and visualize its segmentation results

#create directories for input&output data
img_dir='raw_data'
mkdir $img_dir

res_dir='results'
rect_dir='col_rect'
mkdir $res_dir
mkdir $res_dir/$rect_dir

#download one orginal images from AWS S3
index='pr1954_p246'
AWS_S3_Path="s3://harvardaha-raw-data/personnel-records/1954/scans/firm/$index.png"
aws2 s3 cp $AWS_S3_Path $img_dir

#download correspondent segmentation results from AWS S3
index='pr1954_p0246'    #zero-padded
AWS_S3_Path="s3://harvardaha-results/personnel-records/1954/seg/firm/$rect_dir/${index}_0.json"
aws2 s3 cp $AWS_S3_Path $res_dir/$rect_dir

#####visualize the results
visualization='visualization'
mkdir $res_dir/$visualization

python ../Visualization/RenderBBox.py --imgdir=$img_dir --jsondir=$res_dir/$rect_dir --outputdir=$res_dir/$visualization