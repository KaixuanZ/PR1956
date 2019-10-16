#!/usr/bin/env bash


#remember to generate prob with data from row3
#the current result is for row2, image number/labels are different

InputPath=${InputPath:-'../../results/personnel-records/1956/prob/firm'}

python GraphicalModel_test.py --inputpath=$InputPath



# for teikoku 1957
# 98.15% without graphicalmodel
# 98.36% with manual setup
# 98.56% with automatic estimation