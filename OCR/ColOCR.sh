#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../personnel-records/1956/seg/col_img'}

OutputPath=${OutputPath:-'../../personnel-records/1956/ocr/gcv_output'}

read -p "Do you want to remove previous output in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "removing $OutputPath"

    rm $OutputPath --recursive
fi

python3 ColOCR.py --imgdir=$ImgPath --outputdir=$OutputPath #2>&1 | tee log_Row.txt