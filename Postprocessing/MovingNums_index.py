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
from joblib import Parallel, delayed

def SymbolDetection(mask):
    '''
    :param mask: 1*W mask which denotes if symbol appears on correspondent column of original row_img (H*W)
    :return: intervals of symbols, remove len(symbols)<20
    '''
    list=mask.tolist()
    intervals=[]
    N=len(list)
    for i in range(N):
        if list[i]==0:
            if list[max(i-1,0)]>0:
                intervals.append([i])
        if list[i]>0:
            if list[max(i-1,0)]==0 and intervals:
                intervals[-1].append(i)
    if intervals:   #remove very narrow symbols, which are not characters or numbers
        if len(intervals[-1])==1:
            intervals.pop(-1)

        for i in range(len(intervals)-1,-1,-1):
            if intervals[i][1]-intervals[i][0]<20:
                intervals.pop(i)
    return intervals

def Binarization(img,patchSize=15,threshold=12):
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # local binarization
    img_b = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, patchSize, threshold)
    return img_b

def ExpandRect(rect,threshold=12):
    rect = [list(rect[0]), list(rect[1]), rect[2]]

    if rect[1][0] > rect[1][1]:
        rect[1][1] = rect[1][1] +threshold
    else:
        rect[1][0] = rect[1][0] +threshold

    return tuple(rect)


def main(page,args=None):
    print("processing "+page)
    #read in data
    col_rect_file=os.path.join(args.colrectdir,page+'.json')
    row_rect_file=os.path.join(args.rowrectdir,page+'.json')

    bg_img = cv2.imread('/home/ubuntu/results/personnel-records/1956/seg/background.png')

    with open(col_rect_file) as file:
        col_rects = json.load(file)

    with open(row_rect_file) as file:
        row_rects = json.load(file)

    for key in row_rects.keys():
        row_rects_col = row_rects[key]
        col_img=cv2.imread(os.path.join(args.imgdir,page,page+'_'+key+'.png'))

        col_img_b=Binarization(col_img)
        RLSA_thr=10

        _ , M_col = Rect.CropRect(col_img_b, col_rects[int(key)])
        for i in range(len(row_rects_col)):

            #detect symbols
            row_img_b , _ =Rect.CropRect(col_img_b, Rect.RectOnDstImg(row_rects_col[i],M_col))
            #cv2.imwrite("row_img.png", row_img_b)

            #import pdb;pdb.set_trace()
            count=np.sum(row_img_b/255,axis=0)
            count=signal.medfilt(count, 5)

            _, count=cv2.threshold(count, 5, 255, cv2.THRESH_BINARY_INV)
            #cv2.imwrite("beforeRSLA.png", count)
            count=rlsa.rlsa(count.T, True, False, RLSA_thr)
            #cv2.imwrite("afterRSLA.png", count)

            symbol_intervals=SymbolDetection(count[-1])

            #decide if we need to move symbols closer
            if symbol_intervals:
                if symbol_intervals[-1][0]>0.7*row_img_b.shape[1] and len(symbol_intervals)>1:
                    #expend some area as margin for PIE
                    symbol_intervals[0][0]=max(0,symbol_intervals[0][0]-10)
                    symbol_intervals[-2][1] = symbol_intervals[-2][1] + 10
                    symbol_intervals[-1][0] = symbol_intervals[-1][0] - 10

                    #copy the region of FName (src)
                    row_img, M_col2row = Rect.CropRect(col_img, Rect.RectOnDstImg(ExpandRect(row_rects_col[i]),M_col))
                    src_img=row_img[:,symbol_intervals[0][0]:symbol_intervals[-2][1]].copy()

                    # t is the distance between first and last name
                    t = symbol_intervals[-1][0] - symbol_intervals[-2][1]
                    M_row2col = np.linalg.inv(M_col2row)

                    # manually setup mask, for better performance we should automatically find a mask (binarization,DP,etc)
                    roi_pts = np.array([[0, 0],
                            [src_img.shape[1], 0],
                            [src_img.shape[1], src_img.shape[0]],
                            [0, src_img.shape[0]]], dtype="float32")

                    # mask w.r.t M_row2col
                    roi_pts = Rect.PtsOnDstImg(roi_pts, M_row2col)
                    roi_pts = roi_pts - np.min(roi_pts,axis=0)
                    height,width = np.max(roi_pts, axis=0)[::-1]
                    mask = np.zeros([min(height,src_img.shape[0]),min(width,src_img.shape[1])])
                    roi_mask=cv2.fillConvexPoly(mask, roi_pts, 255)

                    # fill the region of FName with random sampled background
                    center = [[(symbol_intervals[0][0]+symbol_intervals[-2][1]) / 2, row_img.shape[0] / 2]]
                    center = tuple(Rect.PtsOnDstImg(center,M_row2col,False)[-1])
                    x , y = np.random.randint(bg_img.shape[0]-roi_mask.shape[0],size=1)[0], np.random.randint(bg_img.shape[1]-roi_mask.shape[1],size=1)[0]
                    try:
                        #import pdb;pdb.set_trace()
                        col_img_tmp = cv2.seamlessClone(bg_img[x:x+roi_mask.shape[0],y:y+roi_mask.shape[1]], col_img, roi_mask.astype(np.uint8), center, cv2.NORMAL_CLONE)

                        #paste the src region to target region
                        center = [[center[0] + t, row_img.shape[0] / 2]]
                        center = tuple(Rect.PtsOnDstImg(center, M_row2col, False)[-1])
                        col_img = cv2.seamlessClone(src_img, col_img_tmp, roi_mask.astype(np.uint8), center, cv2.NORMAL_CLONE)
                    except:

                        # get error if part of src img is out of dst image
                        # compute on original image can avoid this problem, but this is much faster and there is no big difference
                        print("skip PIE for row "+page+'_'+key +'_'+str(i))

        if not os.path.isdir(os.path.join(args.outputdir,page)):
            os.mkdir(os.path.join(args.outputdir,page))
            print('creating directory ' + os.path.join(args.outputdir,page))
        cv2.imwrite(os.path.join(args.outputdir,page,page+'_'+key+'.png'),col_img)

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Page Detection')
    parser.add_argument('--imgdir', type=str)
    parser.add_argument('--colrectdir', type=str)
    parser.add_argument('--rowrectdir', type=str)
    parser.add_argument('--outputdir', type=str)
    args = parser.parse_args()

    #create output file
    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
        print('creating directory ' + args.outputdir)

    clean_names = lambda x: [i for i in x if i[0] != '.']

    pages = sorted(clean_names(os.listdir(args.imgdir)))
    #import pdb;pdb.set_trace()
    Parallel(n_jobs=1)(map(delayed(main), pages,[args]*len(pages)))
