#!/usr/bin/env bash

ImgDir=${ImgDir:-'../../data/'}

JsonDir=${JsonDir:-'../../output/'}

OutputDir=${OutputDir:-'../../Visualization'}

rm $OutputPath --recursive

python3 Visualization.py --imgdir=$ImgDir --jsondir=$JsonDir --outputdir=$OutputDir