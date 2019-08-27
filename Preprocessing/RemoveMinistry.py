import json
import cv2
import numpy as np
from scipy import signal
import os
from joblib import Parallel, delayed
import argparse
import multiprocessing
import sys
sys.path.append('../')
import Rect

# read image and detected bounding box, output the image with bounding box
def GetColHeight(rect):
    if rect[1][0]>rect[1][1]:
        return rect[1][0]
    else:
        return rect[1][1]

def Binarization(img,patchSize=15,threshold=12):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # local binarization
    img_b = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, patchSize, threshold)
    return img_b

def GetImgFilename(jsonfile):
    book, f, n , p ,c = jsonfile.split('.')[0].split('_')
    f = f[0] + str(int(f[1:]))
    return book + '_' + f + '_' + n + '.tif'

def RemoveMinistry(img,colRects,colJsonNames):
    rect=Rect.CombineRects(colRects[0],colRects[1])
    ROI=[list(rect[0]),list(rect[1]),rect[2]]
    if ROI[1][0]>ROI[1][1]: #divide width by 4
        ROI[1][1]/=4
    else:
        ROI[1][0]/=4
    img_b=Binarization(img,patchSize=31,threshold=10)
    warped_b , _ =Rect.CropRect(img_b, ROI)
    warped_b=cv2.medianBlur(warped_b, 3)
    warped_b=cv2.medianBlur(warped_b, 3)
    kernel = np.ones([21, 3], np.uint8)
    warped_b = cv2.morphologyEx(warped_b, cv2.MORPH_CLOSE, kernel)
    #use CCL to detected largest region
    ret, labels = cv2.connectedComponents(warped_b)
    size , index = 0 , -1
    Height = lambda x: max(x) - min(x)
    for i in range(1, ret + 1):  # O(n^3)
        if labels[labels == i].shape[0] > warped_b.shape[0]*3 and Height(np.where(labels == i)[0])>size:  # remove small CCL regions
            size , index = Height(np.where(labels == i)[0]) , i
    HRange, _ = np.where(labels == index)
    if min(HRange) > 0.05 * warped_b.shape[0] and min(HRange) < 0.3 * warped_b.shape[0]:
        for i in range(2):
            H, theta=min(HRange), colRects[i][2]
            colRect=[list(colRects[i][0]),list(colRects[i][1]),colRects[i][2]]
            if colRect[1][0]>colRect[1][1]:
                colRect[1][0]-=H
            else:
                colRect[1][1]-=H
            if theta < -45:
                theta+=90
            colRect[0]=[colRect[0][0]-H*np.sin(np.deg2rad(theta))/2,colRect[0][1]+H*np.cos(np.deg2rad(theta))/2]
            colRects[i]=colRect
        return True
    return False

def main(coldir,imgdir):
    print("processing "+coldir)
    clean_names = lambda x: [i for i in x if i[0] != '.']
    colRectJsons = sorted(clean_names(os.listdir(coldir)))

    imgpath = os.path.join(imgdir,GetImgFilename(colRectJsons[0]))
    img = cv2.imread(imgpath)

    colRects,colJsonNames = [],[]
    for colRectJson in colRectJsons:
        with open(os.path.join(coldir,colRectJson)) as file:
            colRects.append(json.load(file))
            colJsonNames.append(colRectJson)

    if len(colRects)==5 and abs(GetColHeight(colRects[1])-GetColHeight(colRects[2]))<1:   #haven't remove Ministry
        if RemoveMinistry(img,colRects,colJsonNames):
            print("remove Minsitry for "+colJsonNames[0]+" and "+colJsonNames[1])
            with open(os.path.join(coldir, colJsonNames[0]),'w') as file:
                json.dump(colRects[0], file)
            with open(os.path.join(coldir, colJsonNames[1]),'w') as file:
                json.dump(colRects[1], file)

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Page Detection')
    parser.add_argument('--imgdir', type=str)
    parser.add_argument('--coldir', type=str)
    args = parser.parse_args()

    clean_names = lambda x: [i for i in x if i[0] != '.']
    coldir = os.listdir(args.coldir)
    #coldir = coldir[130::]
    coldir = sorted(clean_names(coldir))

    coldir = [os.path.join(args.coldir, dir) for dir in coldir]
    imgdir = [args.imgdir] * len(coldir)

    #Parallel(n_jobs=1)(map(delayed(main), coldir, imgdir))
    Parallel(n_jobs=multiprocessing.cpu_count())(map(delayed(main), coldir, imgdir))

