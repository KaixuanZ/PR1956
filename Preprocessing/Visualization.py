import json
import cv2
import numpy as np

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

img=cv2.imread('../../data/pr1956_f115_0.tif')
with open('../../output/pr1956_f0115_0_1.json') as jsonfile:
    rect = json.load(jsonfile)

box = cv2.boxPoints(tuple(rect))
box = np.int0(box)
cv2.drawContours(img, [box], 0, (0,0,255), 5)

cv2.imwrite('../../output/tmp1.png',img)