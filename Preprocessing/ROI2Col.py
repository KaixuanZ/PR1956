import json
import cv2
import numpy as np
import os
from joblib import Parallel, delayed
import argparse
import multiprocessing
import sys
sys.path.append('../')
import Rect

#input original scanned img and ROI (text region) bbox, output column bbox

def GetImgFilename(jsonfile):
    #import pdb;pdb.set_trace()
    book, p, sp = jsonfile.split('.')[0].split('_')
    p = p[0] + str(int(p[1:]))
    return book + '_' + p  + '.png'

def ColIndex(img_b,ColWHRatio=2/15):
    '''
    :param img_b: binarized ROI image
    :param ColWHRatio: estimated width and height ratio for column
    :return: col segmentation point
    '''
    theta = range(-90, 271, 24)
    H,W=img_b.shape

    fgPerCol=np.sum(img_b,axis=0)
    fgPerCol = np.convolve(fgPerCol, np.sin(np.deg2rad(theta)))
    fgPerCol = fgPerCol[int(len(theta) / 2):]

    index=[0]
    i=1
    while H*ColWHRatio*(i+0.5)<W:
        #search for vertical line
        l,r=int(H*ColWHRatio*(i-0.25)),int(H*ColWHRatio*(i+0.25))
        tmp=fgPerCol[l:r]
        index.append(l+np.argmax(tmp))
        i+=1
    index.append(W)
    colWidth=np.median([index[i+1]-index[i] for i in range(len(index)-1)])
    index[0]=int(index[1]-colWidth)
    index[-1]=int(index[-2]+colWidth)
    return index

def main(ROIfilename,imgdir,ROIdir,outputdir):
    '''
    :param ROIfilename:
    :param imgdir:
    :param ROIdir:
    :param outputdir:
    :return: rect(s) of columns
    seperate ROI into several columns
    '''
    threshold1=10  #threshold for binarization
    threshold2=190 #threshold for deciding if the last col has content
    print("processing "+ROIfilename)

    imgfilename=GetImgFilename(ROIfilename)
    img = cv2.imread(os.path.join(imgdir,imgfilename), 0)

    with open(os.path.join(ROIdir,ROIfilename)) as file:
        rect = json.load(file)

    warped, M = Rect.CropRect(img, rect)

    #local binarization
    warped_b = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, threshold1)

    H,W=warped_b.shape

    # col index to segment the ROI
    colIndex = ColIndex(warped_b)
    col_rects=[]
    for i in range(len(colIndex) - 1):
        col = warped_b[:, colIndex[i] + int(W / 50):colIndex[i + 1] - int(W / 50)]
        if i<len(colIndex) - 2 or np.max(np.sum(col, axis=0)) / 255 > threshold2:   #check if last col is empty
            box_col=[[colIndex[i],0],[colIndex[i],H-1],[colIndex[i+1],0],[colIndex[i+1],H-1]]

            #get the rect of box_col in original image
            col_rects.append(Rect.RectOnSrcImg(box_col, M))

    #save the rect as json
    outputpath=os.path.join(outputdir, ROIfilename.split('.')[0]+'.json')
    with open(outputpath, 'w') as outfile:
        json.dump(col_rects, outfile)
        print('writing results to ' + outputpath)


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Page Detection')
    parser.add_argument('--imgdir', type=str)
    parser.add_argument('--ROIdir', type=str)
    parser.add_argument('--outputdir', type=str)
    args = parser.parse_args()

    #create output file
    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
        print('creating directory ' + args.outputdir)

    clean_names = lambda x: [i for i in x if i[0] != '.']
    ROIfilenames = os.listdir(args.ROIdir)
    ROIfilenames = sorted(clean_names(ROIfilenames))

    #pagefilenames = pagefilenames[50:]  #start processing at last checkpoint
    imgdir = [args.imgdir] * len(ROIfilenames)
    ROIdir = [args.ROIdir] * len(ROIfilenames)
    outputdir = [args.outputdir] * len(ROIfilenames)

    Parallel(n_jobs=-1)(map(delayed(main), ROIfilenames,imgdir,ROIdir,outputdir))