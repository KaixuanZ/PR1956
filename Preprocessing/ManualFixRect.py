import json
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
import Rect

#input original image and page bbox, output ROI (text region) bbox

def ExpandColHeight(rect,n):
    #scale height by n
    rect = [list(rect[0]), list(rect[1]), rect[2]]

    if rect[1][0] < rect[1][1]:
        rect[1][1] = rect[1][1] * n
    else:
        rect[1][0] = rect[1][0] * n

    return tuple(rect)

def ExpandColWidth(rect,n):
    #scale width by n
    rect = [list(rect[0]), list(rect[1]), rect[2]]

    if rect[1][0] > rect[1][1]:
        rect[1][1] = rect[1][1] * n
    else:
        rect[1][0] = rect[1][0] * n

    return tuple(rect)

def ShiftCol(rect,n):
    #shift center by n (col width)
    rect = [list(rect[0]), list(rect[1]), rect[2]]

    #colWidth=min(rect[1])/5
    colWidth = max(rect[1])

    if rect[2]<-45:
        rect[0][0] = rect[0][0] - colWidth * n * np.cos(np.deg2rad(rect[2]))
        rect[0][1] = rect[0][1] - colWidth * n * np.sin(np.deg2rad(rect[2]))
    else:
        rect[0][0] = rect[0][0] - colWidth * n * np.sin(np.deg2rad(rect[2]))
        rect[0][1] = rect[0][1] - colWidth * n * np.cos(np.deg2rad(rect[2]))

    return rect


pagedir='../../results/personnel-records/1956/seg/supplement/col_rect_wo_ministry'
pagefilename="pr1956_f0184_2_1.json"
with open(os.path.join(pagedir,pagefilename)) as file:
    print("processing "+os.path.join(pagedir,pagefilename))
    rects = json.load(file)

index=[0,1,2,3]
for i in index:
    rect=rects[i]
    t=0.88
    rect=ShiftCol(rect,(1-t)/2)
    rect=ExpandColHeight(rect,t)
    rects[i]=rect

with open(os.path.join(pagedir,pagefilename), 'w') as outfile:
    json.dump(rects, outfile)
    print("writing to " + os.path.join(pagedir, pagefilename))