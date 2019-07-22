import json
import cv2
import numpy as np
import os
from joblib import Parallel, delayed
import argparse

#read image and detected bounding box, output the image with bounding box

def order_points(pts):
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
    box = order_points(box)
    # get width and height of the detected rectangle
    if rect[2]<-45:
        height,width = int(rect[1][0]),int(rect[1][1])
    else:
        width,height = int(rect[1][0]),int(rect[1][1])

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
    return warped

def TifFile(jsonfile):
    jsonfile = jsonfile.split('.')[0]
    book, f, n ,_ = jsonfile.split('_')
    f = f[0] + str(int(f[1:]))   #zeropadding
    n = n
    return book + '_' + f + '_' + n + '.tif'

def main(jsonfile,jsondir,imgdir,outputdir):
    print("processing ", jsonfile)

    scale=4

    img=cv2.imread(os.path.join(imgdir,TifFile(jsonfile)))

    with open(os.path.join(jsondir,jsonfile)) as file:
        rect = json.load(file)

    box = cv2.boxPoints(tuple(rect))
    box = np.int0(box/scale)
    img=cv2.pyrDown(cv2.pyrDown(img))

    cv2.drawContours(img, [box], 0, (0,0,255), 5)

    cv2.imwrite(os.path.join(outputdir,jsonfile.split('.')[0]+'.png'),img)

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Page Detection')
    parser.add_argument('--jsondir', type=str)
    parser.add_argument('--imgdir', type=str)
    parser.add_argument('--outputdir', type=str)
    args = parser.parse_args()

    #create output file
    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
        print('creating directory ' + args.outputdir)

    clean_names = lambda x: [i for i in x if i[0] != '.']
    jsonfile = os.listdir(args.jsondir)
    jsonfile = sorted(clean_names(jsonfile))
    jsondir=[args.jsondir] * len(jsonfile)
    imgdir=[args.imgdir] * len(jsonfile)
    outputdir=[args.outputdir] * len(jsonfile)

    Parallel(n_jobs=36)(map(delayed(main), jsonfile,jsondir,imgdir,outputdir))
