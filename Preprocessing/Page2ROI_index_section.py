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

def main(pagefilename,args):
    '''
    :return: rect(s) of detected ROI
    estimate the ROI by finding the vertical lines
    '''
    #if "pr1956_f0006_2_1" not in pagefilename:
    #    return 0
    print("processing "+pagefilename)
    imgfilename='_'.join(pagefilename.split('_')[:-1])+'.png'

    img = cv2.imread(os.path.join(args.inputdir,imgfilename), 0)

    with open(os.path.join(args.pagedir,pagefilename)) as file:
        rect = json.load(file)

    warped, M_scan2page = Rect.CropRect(img, rect)

    img = cv2.pyrDown(cv2.pyrDown(warped))
    scale = 2 ** 2
    # remove salt-and-pepper noise, reduce the number of CCL areas
    img = cv2.medianBlur(img, 3)
    img = cv2.medianBlur(img, 3)
    # local binarization
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    #cv2.imwrite('binary1.png', img)
    # filling small holes on vertical lines
    kernel = np.ones([5, 5], np.uint8)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # CCL
    ret, labels = cv2.connectedComponents(img)  # CCL
    # find candidate of the four vertical lines
    for i in range(1, ret):  # O(n^3), that's why we need downsampling
        HRange, WRange = np.where(labels == i)
        if (max(HRange) - min(HRange)) > 0.6 * img.shape[0] and (max(WRange) - min(WRange)) > 0.6 * img.shape[1]:
            mask = (labels == i).astype(np.uint8)
            #kernel = np.ones([3, 3], np.uint8)
            #mask= cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            kernel = np.ones([3, 11], np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            #cv2.imwrite('binary2.png', mask)
            mask = mask.astype(np.uint8)
            h_mid ,w_mid=int(img.shape[0]/2),int(mask.shape[1]/2)
            if np.sum(mask[h_mid-100:h_mid+100,w_mid-100:w_mid+100])>0:
                break

        #import pdb;pdb.set_trace()
    if i==ret:
        print("no output for "+ pagefilename)
        return 0
    _ , cnts , _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lines = np.concatenate(cnts, axis=0)
    # fit a rect that include the lines
    rect = cv2.minAreaRect(lines)


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