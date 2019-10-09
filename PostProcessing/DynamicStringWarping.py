import numpy as np
import csv

def DSW(source,target):
    dtw_table = np.zeros([len(target) + 1, len(source) + 1])

    # initialization
    dtw_table[0, :] = np.array([i for i in range(dtw_table.shape[1])])
    dtw_table[:, 0] = np.array([i for i in range(dtw_table.shape[0])])

    # forward update
    for x in range(1, dtw_table.shape[0]):
        for y in range(1, dtw_table.shape[1]):
            if GT[x - 1] == cls[y - 1]:
                dtw_table[x, y] = min(dtw_table[x - 1, y - 1], dtw_table[x, y - 1])
            else:
                dtw_table[x, y] = min(dtw_table[x - 1, y - 1], dtw_table[x, y - 1]) + 1

    # backtracking
    mapping=[]
    return dtw_table[-1,-1],mapping

def ReadCsvCls(filename):
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        cls=''
        for row in spamreader:
            cls+=row[0].split(',')[7][0]
    return cls[1:]


filename='test.csv'
GT='cavptv'
cls=ReadCsvCls(filename)
loss,_=DSW(cls,GT)
print(loss)

filename='test_missing_company.csv'
GT='cavptv'
cls=ReadCsvCls(filename)
loss,_=DSW(cls,GT)
print(loss)