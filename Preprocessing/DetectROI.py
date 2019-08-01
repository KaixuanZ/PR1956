import json
import cv2
import numpy as np
import os
from joblib import Parallel, delayed
import argparse

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


def ExpandCol(rect,n):
    if n>1:
        rect = [list(rect[0]), list(rect[1]), rect[2]]
        if rect[1][0] > rect[1][1]:
            rect[1][1] = rect[1][1] * (n+1) / (n-1)
        else:
            rect[1][0] = rect[1][0] * (n+1) / (n-1)
    else:
        pass
    return tuple(rect)

def GetImgFilename(jsonfile):
    book, f, n , p = jsonfile.split('.')[0].split('_')
    f = f[0] + str(int(f[1:]))
    return book + '_' + f + '_' + n + '.tif'

def main(pagefilename,imgdir,pagedir,outputdir):
    print("processing "+pagefilename)
    imgfilename=GetImgFilename(pagefilename)
    img = cv2.imread(os.path.join(imgdir,imgfilename), 0)

    with open(os.path.join(pagedir,pagefilename)) as file:
        rect = json.load(file)

    warped, M = CropRect(img, rect)

    # local binarization

    warped = cv2.GaussianBlur(warped, (9, 9), 0)
    warped = cv2.pyrDown(warped)
    scale = 2 ** 1
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # CCL
    ret, labels = cv2.connectedComponents(warped)  # CCL
    features = {}

    for i in range(1, ret + 1):  # O(n^3), that's why we need downsampling
        if labels[labels == i].shape[0] > warped.shape[0]:  # remove words (small CCL regions)
            HRange, WRange = np.where(labels == i)
            if (max(HRange) - min(HRange)) > 0.5 * warped.shape[0] and (max(HRange) - min(HRange)) / (
                    max(WRange) - min(WRange)) > 2:
                w = (max(WRange) + min(WRange)) / 2
                features[i] = min(w, warped.shape[1] - w)

    # find the four lines that are most far away from the two sides (some simple classifier)
    if len(features) > 4:
        features = sorted(features.items(), key=lambda kv: kv[1])
        features = features[-4:]
    else:
        features = sorted(features.items(), key=lambda kv: kv[1])
        if len(features)<4:
            print("warning: less than four vertical lines detected for page "+pagefilename)
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

    box = cv2.boxPoints(tuple(rect))
    box = OrderPoints(box) * scale

    box = np.concatenate((box, np.ones([4, 1])), axis=1)
    M_inv = np.linalg.inv(M)
    box = np.dot(M_inv[0:2, :], box.T).T
    box = np.int0(box + 0.5)

    #save the rect as json
    rect=cv2.minAreaRect(box)
    with open(os.path.join(outputdir, pagefilename), 'w') as outfile:
        json.dump(rect, outfile)
        print('writing results to ' + os.path.join(outputdir, pagefilename))


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Page Detection')
    parser.add_argument('--imgdir', type=str)
    parser.add_argument('--pagedir', type=str)
    parser.add_argument('--outputdir', type=str)
    args = parser.parse_args()

    #create output file
    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
        print('creating directory ' + args.outputdir)

    clean_names = lambda x: [i for i in x if i[0] != '.']
    pagefilenames = os.listdir(args.pagedir)
    pagefilenames = sorted(clean_names(pagefilenames))
    pagefilenames = pagefilenames[50:]  #start processing at last checkpoint
    imgdir = [args.imgdir] * len(pagefilenames)
    pagedir = [args.pagedir] * len(pagefilenames)
    outputdir = [args.outputdir] * len(pagefilenames)

    Parallel(n_jobs=7)(map(delayed(main), pagefilenames,imgdir,pagedir,outputdir))