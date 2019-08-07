import json
import cv2
import numpy as np
import os
from joblib import Parallel, delayed
import argparse
import multiprocessing

# read image and detected bounding box, output the image with bounding box

def OrderPoints(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    return np.array([tl, tr, br, bl])


def CropRect(img, rect):
    box = cv2.boxPoints(tuple(rect))
    box = OrderPoints(box)
    # get width and height of the detected rectangle
    if rect[2] < -45:
        height, width = int(rect[1][0]), int(rect[1][1])
    else:
        width, height = int(rect[1][0]), int(rect[1][1])

    src_pts = box.astype("float32")
    # corrdinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped, M

def GetImgFilename(jsonfile):
    book, f, n , p = jsonfile.split('.')[0].split('_')
    f = f[0] + str(int(f[1:]))
    return book + '_' + f + '_' + n + '.tif'

def ColIndex(img_b,ColWHRatio=2/15):
    theta = range(-90, 271, 30)
    H,W=img_b.shape

    fgPerCol=np.sum(img_b,axis=0)
    fgPerCol = np.convolve(fgPerCol, np.sin(np.deg2rad(theta)))
    fgPerCol = fgPerCol[int(len(theta) / 2):]

    index=[0]
    i=1
    while i<=4:
        if  H*ColWHRatio*(i+0.5)<W:
            #search for vertical line
            l,r=int(H*ColWHRatio*(i-0.25)),int(H*ColWHRatio*(i+0.25))
            tmp=fgPerCol[l:r]
            index.append(l+np.argmax(tmp))
            i+=1
        else:
            break
    index.append(W)
    return index

def main(ROIfilename,imgdir,ROIdir,outputdir):
    threshold1=10  #threshold for binarization
    threshold2=100 #threshold for
    print("processing "+ROIfilename)
    outputdir=os.path.join(outputdir,ROIfilename.split('.')[0])
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
        print('creating directory ' + outputdir)

    imgfilename=GetImgFilename(ROIfilename)
    img = cv2.imread(os.path.join(imgdir,imgfilename), 0)

    with open(os.path.join(ROIdir,ROIfilename)) as file:
        rect = json.load(file)

    warped, M = CropRect(img, rect)
    M_inv = np.linalg.inv(M)

    #local binarization
    warped_b = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, threshold1)

    H,W=warped_b.shape
    colIndex = ColIndex(warped_b)
    for i in range(len(colIndex) - 1):
        col = warped_b[:, colIndex[i] + int(warped_b.shape[1] / 100):colIndex[i + 1] - int(warped_b.shape[1] / 100)]
        if i<len(colIndex) - 2 or np.max(np.sum(col, axis=0)) / 255 > threshold2:
            box_col=[[colIndex[i],0],[colIndex[i],H],[colIndex[i+1],0],[colIndex[i+1],H]]
            box_col=OrderPoints(np.array(box_col))

            #get the location of box_col in original image
            box_col = np.concatenate((box_col, np.ones([4, 1])), axis=1)
            box_col = np.dot(M_inv[0:2, :], box_col.T).T
            box_col = np.int0(box_col + 0.5)

            #save the rect as json
            rect=cv2.minAreaRect(box_col)
            outputpath=os.path.join(outputdir, ROIfilename.split('.')[0]+'_'+str(i)+'.json')
            with open(outputpath, 'w') as outfile:
                json.dump(rect, outfile)
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

    Parallel(n_jobs=multiprocessing.cpu_count())(map(delayed(main), ROIfilenames,imgdir,ROIdir,outputdir))