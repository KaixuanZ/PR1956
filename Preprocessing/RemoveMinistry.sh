#!/usr/bin/env bash

ImgPath=${ImgPath:-'../../personnel-records/1956/scans/parsed'}

ColPath=${ROIPath:-'../../personnel-records/1956/seg/col_rect'}

python3 RemoveMinistry.py --imgdir=$ImgPath --coldir=$ColPath  #2>&1 | tee log_Row.txt