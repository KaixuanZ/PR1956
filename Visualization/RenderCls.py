import json
import cv2
import numpy as np
import os
import Rect
import argparse
from joblib import Parallel, delayed

clean_names = lambda x: [i for i in x if i[0] != '.']

#255-[b,g,r] because we want to output img with white background
ColorDict={"company":[255-0,255-0,255-255],  #red
            "address":[255-0,255-128,255-0],    #green
            "variable":[255-32,255-165,255-218],        #golden rod
            "value":[255-209,255-206,255-0],    #dark turquoise
            "personnel":[255-205,255-0,255-0],  #medium blue
            "table":[255-139,255-0,255-139]}    #dark magenta

def Binarization(img,patchSize=9,threshold=7):
    if len(img.shape)>1:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    # local binarization
    img_b = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, patchSize, threshold)
    return img_b

def GetImgFilename(jsonfile):
    book, p, _ = jsonfile.split('.')[0].split('_')
    p = p[0] + str(int(p[1:]))
    return book + '_' + p + '.png'

def main(ROIrect,args):
    scale = 2
    img = cv2.imread(os.path.join(args.imgdir,GetImgFilename(ROIrect)),0)
    img = cv2.pyrDown(img)
    img_b = Binarization(img)*255

    H,W = img_b.shape
    res = np.zeros([H,W,3], np.uint8)


    with open(os.path.join(args.clsdir,ROIrect)) as clsjson:
        cls = json.load(clsjson)
    cls=cls["name"]

    with open(os.path.join(args.rowrectdir, ROIrect)) as rectjson:
        rowrects = json.load(rectjson)
    tmp=[]
    for key in rowrects.keys():
        tmp+=rowrects[key]
    rowrects=tmp

    with open(os.path.join(args.colrectdir, ROIrect)) as rectjson:
        ROIrect = json.load(rectjson)

    for i in range(len(rowrects)):

        box = cv2.boxPoints(tuple(rowrects[i]))
        box = np.int0(box/scale)
        mask = np.zeros([H,W,3], np.uint8)

        cv2.drawContours(mask, [box], -1, ColorDict[cls[i]], -1)
        b,g,r = mask[:,:,0],mask[:,:,1],mask[:,:,2]
        #import pdb;pdb.set_trace()
        b,g,r = b*img_b,g*img_b,r*img_b

        mask = cv2.merge([b,g,r])
        #import pdb;pdb.set_trace()


        res+=mask

        res=255-res

    box = cv2.boxPoints(tuple(ROIrect))
    box = np.int0(box/scale)
    ROI_rect = cv2.minAreaRect(box)
    warped,_=Rect.CropRect(res,ROI_rect)

    cv2.imwrite(os.path.join(args.outputdir,ROIrect.split('.')[0]+'.png'),warped)

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Visualize classification results')
    parser.add_argument( '--imgdir', type=str)
    parser.add_argument( '--outputpath', type=str)
    parser.add_argument( '--pagerectdir', type=str)
    parser.add_argument( '--rowrectdir', type=str)
    parser.add_argument( '--clsdir', type=str)

    args = parser.parse_args()
    clean_names = lambda x: [i for i in x if i[0] != '.']
    ROIrects=sorted(clean_names(os.listdir(args.pagerectdir)))

    Parallel(n_jobs=-1)(map(delayed(main), ROIrects, [args] * len(ROIrects)))