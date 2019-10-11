import os
import cv2
from joblib import Parallel, delayed


def CopyAndPaste(file,inputdir,outputdir):
    print("processing " + file)
    inputpath = os.path.join(inputdir, file)
    filename = file.split('_')
    filename[1] = filename[1][0] + filename[1][1:].zfill(4)
    outputpath = os.path.join(outputdir, '_'.join(filename))
    cv2.imwrite(outputpath, cv2.imread(inputpath))

inputdir='../raw_data/personnel-records/1956/scans/parsed'
outputdir='../raw_data/personnel-records/1956/scans/firm'
clean_names = lambda x: [i for i in x if i[0] != '.']
files=sorted(clean_names(os.listdir(inputdir)))


Parallel(n_jobs=-1)(map(delayed(CopyAndPaste), files, [inputdir]*len(files),[outputdir]*len(files)))