import json
import cv2
import numpy as np
import os
import Rect

clean_names = lambda x: [i for i in x if i[0] != '.']

#255-[b,g,r] because we want to output img with white background
ColDict={"company":[255-0,255-0,255-255],  #red
            "address":[255-0,255-128,255-0],    #green
            "variable":[255-32,255-165,255-218],        #golden rod
            "value":[255-209,255-206,255-0],    #dark turquoise
            "personnel":[255-205,255-0,255-0],  #medium blue
            "table":[255-139,255-0,255-139]}    #dark magenta

def Binarization(img,patchSize=9,threshold=7):
    if len(img.shape)>1:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    # local binarization
    img_b = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, patchSize, threshold)
    return img_b

outputdir='visualization'
jsondir='Row_rect'
files=sorted(clean_names(os.listdir(jsondir)))
scale = 2
#import pdb;pdb.set_trace()
img = cv2.imread('pr1956_f115_4.tif')
img = cv2.pyrDown(img)
img_b = Binarization(img)*255

H,W = img_b.shape
res = np.zeros([H,W,3], np.uint8)
with open("cls.json") as clsjson:
    cls = json.load(clsjson)
cls=cls["name"]

for i in range(58+59,58+59+74):
    if i%20==0:
        print(i)
    with open(os.path.join(jsondir,files[i])) as rectjson:
        rect = json.load(rectjson)
    box = cv2.boxPoints(tuple(rect))
    box = np.int0(box/scale)
    mask = np.zeros([H,W,3], np.uint8)

    cv2.drawContours(mask, [box], -1, ColDict[cls[i]], -1)
    b,g,r = mask[:,:,0],mask[:,:,1],mask[:,:,2]
    #import pdb;pdb.set_trace()
    b,g,r = b*img_b,g*img_b,r*img_b

    mask = cv2.merge([b,g,r])
    #import pdb;pdb.set_trace()


    res+=mask

res=255-res

with open('Col_rect/pr1956_f0115_4_0_2.json') as rectjson:
    ROI_rect = json.load(rectjson)
box = cv2.boxPoints(tuple(ROI_rect))
box = np.int0(box/scale)
ROI_rect = cv2.minAreaRect(box)
warped,_=Rect.CropRect(res,ROI_rect)

cv2.imwrite(os.path.join(outputdir,'Cls.png'),warped)
