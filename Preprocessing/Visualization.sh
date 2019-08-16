#!/usr/bin/env bash

ImgDir=${ImgDir:-'../../personnel-records/1956/scans/parsed'}

JsonDir=${JsonDir:-'../../personnel-records/1956/sge/row_rect'}

OutputDir=${OutputDir:-'../../personnel-records/1956/visualization'}

echo "removing $OutputDir"

rm -rf $OutputDir

python3 Visualization.py --imgdir=$ImgDir --jsondir=$JsonDir --outputdir=$OutputDir