import numpy as np
import csv

def DSW(source,target):
    '''
    :param source
    :param target
    :return: [loss,mapping]
            loss represents the difference between source and target (block structure)
            mapping means how to map target to source and get minimum loss
    '''
    dtw_table = np.zeros([len(target) + 1, len(source) + 1])

    # initialization
    dtw_table[0, :] = np.array([i for i in range(dtw_table.shape[1])])
    dtw_table[:, 0] = np.array([i for i in range(dtw_table.shape[0])])

    # forward update
    for x in range(1, dtw_table.shape[0]):
        for y in range(1, dtw_table.shape[1]):
            if target[x - 1] == source[y - 1]:
                dtw_table[x, y] = min(dtw_table[x - 1, y - 1], dtw_table[x, y - 1])
            else:
                dtw_table[x, y] = min(dtw_table[x - 1, y - 1], dtw_table[x, y - 1]) + 1

    index=np.argmin(dtw_table[:,-1])

    # get path
    cls,x,y='',index,len(source)

    for i in range(0,len(source)):
        cls=target[x-1]+cls
        y-=1
        if dtw_table[x,y]>dtw_table[x-1,y]:
            x-=1


    #check if dtw_table[-1,-1] is the largest in dtw_table[:,-1], o.w. it means the structure of this company hasn't been appeared
    return dtw_table[index,-1],cls

def ReadCsvCls(filename):
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        cls=''
        for row in spamreader:
            cls+=row[0].split(',')[7][0]
    return cls[1:]


#filename='test.csv'
filename='test_missing_company.csv'
#'cavptv' or 'cavpxtv' x stands for variable name (accounting period) only one row
# also we can only correct the 'vptv'/'vpxtv' part (should only return one v at beginning and one v at last (maybe another one for accounting period))

GT='cavptv'
cls=ReadCsvCls(filename)
cost,res=DSW(cls,GT)
print("cost value: \t",cost)
print("block structure: \t",GT)
print("original classification: \t",cls)
print("corrected classification: \t",res)