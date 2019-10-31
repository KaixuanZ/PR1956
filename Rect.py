#includes functions related with Rect computation

import numpy as np
import cv2
from shapely.geometry import Polygon

def AreaOfOverlap(rect1,rect2,rect=True):
    #return the size of intersection area of rect1 and rect2
    if rect:
        plg1 = Polygon(cv2.boxPoints(tuple(rect1)))
        plg2 = Polygon(cv2.boxPoints(tuple(rect2)))
    return plg1.intersection(plg2).area

def CombineRects(rect1,rect2):
    #return a minAreaRect which contains rect1 and rect2
    pts1=cv2.boxPoints(tuple(rect1))
    pts2=cv2.boxPoints(tuple(rect2))
    pts=np.concatenate((pts1,pts2),axis=0)
    return cv2.minAreaRect(pts)

def DistOfRects(rect1,rect2):
    #return the L2 distance of centers of two rects
    c1,c2=np.array(rect1[0]),np.array(rect2[0])
    return np.sum((c1-c2)**2)**0.5

def RectOnDstImg(rect, M_src2dst,flag_box=False):
    # Given rect on src img, and transformation M from src to dst, return the rect on dst img.
    if flag_box:
        pts=PtsOnDstImg(rect,M_src2dst)
    else:
        pts=PtsOnDstImg(cv2.boxPoints(tuple(rect)), M_src2dst)
    return cv2.minAreaRect(pts)

def PtsOnDstImg(pts, M_src2dst, orderPts=True):
    #Given pts on src img, and transformation M from src to dst, return the pts on dst img.

    pts = np.array(pts)
    pts = np.concatenate((pts, np.ones([pts.shape[0], 1])), axis=1)
    # pts on the dst img
    pts = np.dot(M_src2dst, pts.T).T
    pts = pts / pts[:, 2, None]
    pts = np.int0(pts[:,0:2]+0.5)
    if orderPts:
        return OrderPoints(pts)
    return pts

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
    pts = cv2.boxPoints(tuple(rect))    # e.g. [[0,0],[0,1],[1,1],[1,0]] <==> ((0.5, 0.5), (1.0, 1.0), -90.0)
    pts = OrderPoints(pts)
    # get width and height of the detected rectangle
    if rect[2] < -45:
        height, width = int(rect[1][0]), int(rect[1][1])
    else:
        width, height = int(rect[1][0]), int(rect[1][1])

    src_pts = pts.astype("float32")
    # corrdinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped=None
    if img is not None:
        warped = cv2.warpPerspective(img, M, (width+1, height+1))
    return warped, M