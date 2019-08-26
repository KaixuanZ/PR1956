import os
import io
import json
from joblib import Parallel, delayed
import argparse
import multiprocessing
import Rect
import cv2

def GetImgFilename(jsonfile):
    book, f, n , p ,c = jsonfile.split('.')[0].split('_')
    f = f[0] + str(int(f[1:]))
    return book + '_' + f + '_' + n + '.tif'

def main(coldir, imgdir, outputdir):
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
        print('creating directory ' + outputdir)

    clean_names = lambda x: [i for i in x if i[0] != '.']
    colRectJsons = sorted(clean_names(os.listdir(coldir)))

    imgpath = os.path.join(imgdir, GetImgFilename(colRectJsons[0]))
    img=cv2.imread(imgpath)

    colRects, colJsonNames = [], []
    for colRectJson in colRectJsons:
        with open(os.path.join(coldir, colRectJson)) as file:
            colRects.append(json.load(file))
            colJsonNames.append(colRectJson)

    for i in range(len(colRects)):
        col, _ = Rect.CropRect(img, colRects[i])
        cv2.imwrite(os.path.join(outputdir,colJsonNames[i].split('.')[0]+'.png'),col)

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Page Detection')
    parser.add_argument('--imgdir', type=str)
    parser.add_argument('--coldir', type=str)
    parser.add_argument('--outputdir', type=str)
    args = parser.parse_args()

    #create output file
    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
        print('creating directory ' + args.outputdir)

    clean_names = lambda x: [i for i in x if i[0] != '.']
    coldir = os.listdir(args.coldir)
    #coldir = coldir[130::]
    coldir = sorted(clean_names(coldir))

    outputdir = [os.path.join(args.outputdir, dir) for dir in coldir]
    coldir = [os.path.join(args.coldir, dir) for dir in coldir]
    imgdir = [args.imgdir] * len(coldir)

    Parallel(n_jobs=1)(map(delayed(main), coldir, imgdir, outputdir))
    #Parallel(n_jobs=multiprocessing.cpu_count())(map(delayed(main), coldir, imgdir, outputdir))

