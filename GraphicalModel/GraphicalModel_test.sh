#!/usr/bin/env bash


#InputPath=${InputPath:-'../../personnel-records/1954/prob/'}

InputPath=${InputPath:-'prob_pr1954/'}

python GraphicalModel.py --inputpath=$InputPath



# for teikoku 1957
# 98.15% without graphicalmodel
# 98.36% with manual setup
# 98.56% with automatic estimation