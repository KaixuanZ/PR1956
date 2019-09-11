#includes functions related with Rect computation

import numpy as np
import cv2
from shapely.geometry import Polygon

def AreaOfOverlap(rect1,rect2,rect=True):
    #return the size of intersection area of rect1 and rect2
    if rect:
        rect1 = Polygon(cv2.boxPoints(tuple(rect1)))
        rect2 = Polygon(cv2.boxPoints(tuple(rect2)))
    return rect1.intersection(rect2).area

def CombineRects(rect1,rect2):
    #return a minAreaRect which contains rect1 and rect2
    box1=cv2.boxPoints(tuple(rect1))
    box2=cv2.boxPoints(tuple(rect2))
    box=np.concatenate((box1,box2),axis=0)
    return cv2.minAreaRect(box)

def DistOfRects(rect1,rect2):
    #return the L2 distance of centers of two rects
    c1,c2=np.array(rect1[0]),np.array(rect2[0])
    return np.sum((c1-c2)**2)**0.5


def RectOnSrcImg(box, M):
    # Given box on dst img, and transformation M from src to dst, return the box on src img.
    box=PtsOnSrcImg(box, M)
    return cv2.minAreaRect(box)

def PtsOnSrcImg(pts, M):
    #Given pts on dst img, and transformation M from src to dst, return the pts on src img.
    pts = np.array(pts)
    pts = np.concatenate((pts, np.ones([pts.shape[0], 1])), axis=1)
    M_inv = np.linalg.inv(M)
    # box on the original img
    pts = np.dot(M_inv[0:2, :], pts.T).T
    pts = np.int0(pts + 0.5)
    return OrderPoints(pts)

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
    #crop img w.r.t. rect, return the warped img and the transformation M (from src to dst)
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
                        [width-1, 0],
                        [width-1, height-1],
                        [0, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped, M