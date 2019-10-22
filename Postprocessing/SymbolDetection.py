import json
import cv2
import numpy as np
import os
import argparse
import sys
sys.path.append('../')
import Rect
from pythonRLSA import rlsa
from scipy import signal

def SymbolDetection(mask,RLSA_thr,margin_thr=10):
    '''
    :param mask: 1*W mask which denotes if symbol appears on correspondent column of original row_img (H*W)
    :param RLSA_thr: threshold for RLSA algorithm
    :return: intervals of symbols connected by RLSA
    '''
    list=[0]+mask.tolist()+[0]
    intervals=[]
    N=len(list)
    for i in range(N):
        if list[i]==0:
            if list[max(i-1,0)]>0:
                intervals[-1].append(i-1)
                if intervals[-1][1]-intervals[-1][0]<=RLSA_thr*0.5:
                    intervals.pop(-1)
                else:
                    intervals[-1]=[max(intervals[-1][0]-margin_thr,0),min(intervals[-1][1]+margin_thr,N-3)]
            if list[min(i+1,N-1)]>0:
                intervals.append([i+1-1])
    return intervals

def Binarization(img,patchSize=15,threshold=12):
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # local binarization
    img_b = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, patchSize, threshold)
    return img_b

def CombineImgs(img_src,img_dst,M_src2dst):
    '''
    combine color images
    :param img_src:
    :param img_dst:
    :param M_src2dst:
    :return:
    '''
    warped = cv2.warpPerspective(img_src, M_src2dst, (img_dst.shape[1], img_dst.shape[0]))
    mask=(warped==0).astype(np.uint8)
    #import pdb;pdb.set_trace()
    img_dst = np.multiply(img_dst,mask)

    return img_dst+warped


cls_file='/home/ubuntu/results/personnel-records/1956/cls/CRF/firm/pr1956_f0046_8_1.json'
col_rect_file='/home/ubuntu/results/personnel-records/1956/seg/firm/col_rect/pr1956_f0046_8_1.json'
row_rect_file='/home/ubuntu/results/personnel-records/1956/seg/firm/row_rect/pr1956_f0046_8_1.json'


with open(cls_file) as file:
    cls = json.load(file)
cls=cls[:44]
with open(col_rect_file) as file:
    col_rects = json.load(file)
with open(row_rect_file) as file:
    row_rects = json.load(file)
tmp=[]
for key in row_rects.keys():
    tmp+=row_rects[key]
row_rects=tmp

col_img=cv2.imread('/home/ubuntu/results/personnel-records/1956/seg/firm/col_img/pr1956_f0046_8_1/pr1956_f0046_8_1_0.png')
col_img_b=Binarization(col_img)
cv2.imwrite('tmp_col.png',col_img)
RLSA_thr=60

_ , M_col = Rect.CropRect(col_img_b, col_rects[0])
for i in range(len(cls)):
    if cls[i]=='personnel':
        #detect symbols
        row_img_b , M_col2row =Rect.CropRect(col_img_b, Rect.RectOnDstImg(row_rects[i],M_col))
        count=np.sum(row_img_b/255,axis=0)
        count=signal.medfilt(count, 5)

        _, count=cv2.threshold(count, 4, 255, cv2.THRESH_BINARY_INV)
        #import pdb;pdb.set_trace()
        count=count.T
        rlsa.rlsa(count, True, False, RLSA_thr)

        symbol_intervals=SymbolDetection(255-count[-1],RLSA_thr)

        #decide if we need to move symbols closer
        #import pdb;pdb.set_trace()
        if symbol_intervals:
            if symbol_intervals[-1][0]>0.6*row_img_b.shape[1]:
                #copy the region of FName (src)
                row_img, _ = Rect.CropRect(col_img, Rect.RectOnDstImg(row_rects[i],M_col))

                src_img=row_img[:,symbol_intervals[-2][0]:symbol_intervals[-2][1]]

                #paste the src region to target region
                t=symbol_intervals[-1][0]-symbol_intervals[-2][1]
                row_img[:, symbol_intervals[-2][0]+t:symbol_intervals[-2][1]+t]=src_img

                # fill the region of FName with background
                #row_img[:, symbol_intervals[-2][0]:symbol_intervals[-2][0]+t]=[0,0,0]

                #put the modified region back to orginal col image
                col_img=CombineImgs(row_img,col_img,np.linalg.inv(M_col2row))
cv2.imwrite('tmp_modified_col.png',col_img)
import pdb;pdb.set_trace()