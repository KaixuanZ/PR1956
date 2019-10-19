import argparse
import os
import json
import numpy as np

#in the cls json file, 2 represent company name
inputpath='../../results/personnel-records/1956/cls/CRF/supplement/'
clean_names = lambda x: [i for i in x if i[0] != '.']
count=0
for dir in sorted(clean_names(os.listdir(inputpath))):  #one dir includes prob for one page
    print("processing "+dir)
    with open(os.path.join(inputpath,dir)) as jsonfile:
        cls = json.load(jsonfile)
    for i in range(len(cls)):
        if i==0 and cls[i]=='company':
            count+=1
        elif cls[i]=='company' and cls[max(i-1,0)]!='company':
            count+=1
print(count)