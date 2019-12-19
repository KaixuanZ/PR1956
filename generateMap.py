import os
import json


section_range={}
section_range['1956']={}
section_range['1956']['firm']=['pr1956_f0046_8','pr1956_f0183_0']
section_range['1956']['bank']=['pr1956_f0041_2','pr1956_f0046_7']
section_range['1956']['credit_union']=['pr1956_f0183_1','pr1956_f0184_1']
section_range['1956']['index']=['pr1956_f0004_1','pr1956_f0007_3']
section_range['1956']['official_office']=['pr1956_f0007_4','pr1956_f0023_4']
section_range['1956']['supplement']=['pr1956_f0184_2','pr1956_f0198_2']

def index2section(index):
    year = index.split('_')[0][2:]
    for section in section_range[year]:
        if index>=section_range[year][section][0] and index<=section_range[year][section][1]:
            return section
    print("error: data doesn't exist")
    return None

def map(index,filetype,prefix='s3://harvardaha-results',book='personnel-records'):
    year = index.split('_')[0][2:]
    section=index2section(index)
    if section:
        path=os.path.join(prefix,book,year,'seg',section,filetype,index)
        if not os.path.exists(path):
            print("warning: " +path+ " doesn't exist")
        return path
    return None

index="pr1956_f0042_0_0"
filetype="col_img"

print(map(index,filetype,prefix='/home/ubuntu/results'))