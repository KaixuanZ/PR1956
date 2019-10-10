import json
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
import Rect

#input original image and page bbox, output ROI (text region) bbox

def ExpandCol(rect,n):
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

    colWidth=min(rect[1])/5

    if rect[2]>-45:
        rect[0][0] = rect[0][0] - colWidth * n * np.cos(np.deg2rad(rect[2]))
        rect[0][1] = rect[0][1] - colWidth * n * np.sin(np.deg2rad(rect[2]))
    else:
        rect[0][0] = rect[0][0] - colWidth * n * np.sin(np.deg2rad(rect[2]))
        rect[0][1] = rect[0][1] - colWidth * n * np.cos(np.deg2rad(rect[2]))

    return rect


pagedir='../../results/personnel-records/1954/seg/official_office/ROI_rect'
pagefilename="pr1954_p0111_1.json"
with open(os.path.join(pagedir,pagefilename)) as file:
    print("processing "+os.path.join(pagedir,pagefilename))
    rect = json.load(file)

rect=ExpandCol(rect,5/6)

#rect=ShiftCol(rect,-0.5)

with open(os.path.join(pagedir,pagefilename), 'w') as outfile:
    json.dump(rect, outfile)
    print("writing to " + os.path.join(pagedir, pagefilename))