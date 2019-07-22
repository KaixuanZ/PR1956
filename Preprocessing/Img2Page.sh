#!/usr/bin/env bash

InputPath=${InputPath:-'../../data/'}

OutputPath=${OutputPath:-'../../output/'}

rm $OutputPath --recursive

python3 Img2Page.py --inputpath=$InputPath --outputpath=$OutputPath 2>&1 | tee log_Img2Page.txt