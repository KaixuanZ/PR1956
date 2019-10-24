import json
import cv2
import numpy as np
import os
from joblib import Parallel, delayed
import argparse
import sys
sys.path.append('../')
import Rect

#input original image and page bbox, output ROI (text region) bbox

def ExpandCol(rect,n):
    rect = [list(rect[0]), list(rect[1]), rect[2]]
    if n>1:
        if rect[1][0] > rect[1][1]:
            rect[1][1] = rect[1][1] * (n+1) / (n-1)
        else:
            rect[1][0] = rect[1][0] * (n+1) / (n-1)
    else:
        if rect[1][0] > rect[1][1]:
            rect[1][1] = rect[1][1] + rect[1][0] * 0.1325 * 2
        else:
            rect[1][0] = rect[1][0] + rect[1][1] * 0.1325 * 2
    return tuple(rect)

def main(pagefilename,args):
    '''
    :return: rect(s) of detected ROI
    estimate the ROI by finding the vertical lines
    '''
    print("processing "+pagefilename)
    imgfilename='_'.join(pagefilename.split('_')[:-1])+'.png'

    img = cv2.imread(os.path.join(args.inputdir,imgfilename), 0)

    with open(os.path.join(args.pagedir,pagefilename)) as file:
        rect = json.load(file)

    warped, M_scan2page = Rect.CropRect(img, rect)

    warped = cv2.pyrDown(warped)
    scale = 2 ** 1
    #remove salt-and-pepper noise, reduce the number of CCL areas
    warped = cv2.medianBlur(warped, 3)
    warped = cv2.medianBlur(warped, 3)
    #local binarization
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    #filling small holes on vertical lines
    kernel = np.ones([7,1], np.uint8)
    warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel)

    # CCL
    ret, labels = cv2.connectedComponents(warped)  # CCL
    features = {}
    #find candidate of the four vertical lines
    for i in range(1, ret + 1):  # O(n^3), that's why we need downsampling
        if labels[labels == i].shape[0] > warped.shape[0]:  # remove words (small CCL regions)
            HRange, WRange = np.where(labels == i)
            if (max(HRange) - min(HRange)) > 0.4 * warped.shape[0] and (max(HRange) - min(HRange)) / (
                    max(WRange) - min(WRange)) > 15 and min(WRange)>0.1*warped.shape[1] and max(WRange)<0.9*warped.shape[1]:
                w = (max(WRange) + min(WRange)) / 2
                features[i] = min(w, warped.shape[1] - w)
    # import pdb;pdb.set_trace()
    # find the four lines that are most far away from the two sides (some simple classifier)
    if len(features) > 4:
        features = sorted(features.items(), key=lambda kv: kv[1])
        features = features[-4:]
    else:
        if len(features)>0:
            features = sorted(features.items(), key=lambda kv: kv[1])
            if len(features)<4:
                print("warning: less than four vertical lines detected for page "+pagefilename)
        else:
            print("warning: no vertical line detected for page " + pagefilename)
            return 0
    index = [item[0] for item in features]

    lines = np.zeros(labels.shape)  # mask for lines
    for i in index:
        lines = lines + (labels == i).astype(int)

    #import pdb;pdb.set_trace()

    _ , cnts , _ = cv2.findContours(lines.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lines = np.concatenate(cnts, axis=0)
    # fit a rect that include the lines
    rect = cv2.minAreaRect(lines)
    rect = ExpandCol(rect,len(features))

    box = cv2.boxPoints(tuple(rect)) * scale

    rect=Rect.RectOnDstImg(box, np.linalg.inv(M_scan2page),True)

    #save the rect as json
    with open(os.path.join(args.outputdir, pagefilename), 'w') as outfile:
        json.dump(rect, outfile)
        #print('writing results to ' + os.path.join(args.outputdir, pagefilename))


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Page Detection')
    parser.add_argument('--inputdir', type=str)
    parser.add_argument('--pagedir', type=str)
    parser.add_argument('--outputdir', type=str)
    args = parser.parse_args()

    #create output file
    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
        print('creating directory ' + args.outputdir)

    clean_names = lambda x: [i for i in x if i[0] != '.']
    pagefilenames = sorted(clean_names(os.listdir(args.pagedir)))

    Parallel(n_jobs=-1)(map(delayed(main), pagefilenames,[args]*len(pagefilenames)))