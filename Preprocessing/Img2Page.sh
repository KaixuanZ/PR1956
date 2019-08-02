#!/usr/bin/env bash

InputPath=${InputPath:-'../../personnel-records/1956/scans/parsed/'}

OutputPath=${OutputPath:-'../../personnel-records/1956/seg/page_rect/'}

#rm $OutputPath --recursive

python3 Img2Page.py --inputpath=$InputPath --outputpath=$OutputPath 2>&1 | tee log_Img2Page.txt

./RemoveAdPage.sh --Path=$OutputPath