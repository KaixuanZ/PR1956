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
    cls_file=os.path.join(args.clsdir,page+'.json')
    col_rect_file=os.path.join(args.colrectdir,page+'.json')
    row_rect_file=os.path.join(args.rowrectdir,page+'.json')

    bg_img = cv2.imread('/home/ubuntu/results/personnel-records/1956/seg/background.png')

    with open(col_rect_file) as file:
        col_rects = json.load(file)

    with open(row_rect_file) as file:
        row_rects = json.load(file)

    with open(cls_file) as file:
        cls = json.load(file)

    for key in row_rects.keys():
        row_rects_col = row_rects[key]
        col_img=cv2.imread(os.path.join(args.imgdir,page,page+'_'+key+'.png'))

        col_img_b=Binarization(col_img)
        RLSA_thr=50

        _ , M_col = Rect.CropRect(col_img_b, col_rects[int(key)])
        for i in range(len(row_rects_col)):
            if cls[i]=='personnel':
                #detect symbols
                row_img_b , _ =Rect.CropRect(col_img_b, Rect.RectOnDstImg(row_rects_col[i],M_col))
                count=np.sum(row_img_b/255,axis=0)
                count=signal.medfilt(count, 5)
                _, count=cv2.threshold(count, 3, 255, cv2.THRESH_BINARY_INV)
                count=rlsa.rlsa(count.T, True, False, RLSA_thr)

                symbol_intervals=SymbolDetection(255-count[-1],RLSA_thr)

                #decide if we need to move symbols closer
                if symbol_intervals:
                    if symbol_intervals[-1][0]>0.6*row_img_b.shape[1] and len(symbol_intervals)>1:
                        #copy the region of FName (src)
                        row_img, M_col2row = Rect.CropRect(col_img, Rect.RectOnDstImg(ExpandRect(row_rects_col[i]),M_col))
                        src_img=row_img[:,symbol_intervals[-2][0]:symbol_intervals[-2][1]].copy()

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
                        center = [[np.median(symbol_intervals[-2]), row_img.shape[0] / 2]]
                        center = tuple(Rect.PtsOnDstImg(center,M_row2col,False)[-1])
                        x , y = np.random.randint(bg_img.shape[0]-roi_mask.shape[0],size=1)[0], np.random.randint(bg_img.shape[1]-roi_mask.shape[1],size=1)[0]
                        try:
                            col_img = cv2.seamlessClone(bg_img[x:x+roi_mask.shape[0],y:y+roi_mask.shape[1]], col_img, roi_mask.astype(np.uint8), center, cv2.NORMAL_CLONE)

                            #paste the src region to target region
                            center = [[np.median(symbol_intervals[-2]) + t, row_img.shape[0] / 2]]
                            center = tuple(Rect.PtsOnDstImg(center, M_row2col, False)[-1])
                            col_img = cv2.seamlessClone(src_img, col_img, roi_mask.astype(np.uint8), center, cv2.NORMAL_CLONE)
                        except:
                            # get error if part of src img is out of dst image
                            # compute on original image can avoid this problem, but this is much faster and there is no big difference
                            print("ignore first/last row for "+page+'_'+key )
        cls=cls[len(row_rects_col):]

        if not os.path.isdir(os.path.join(args.outputdir,page)):
            os.mkdir(os.path.join(args.outputdir,page))
            print('creating directory ' + os.path.join(args.outputdir,page))
        cv2.imwrite(os.path.join(args.outputdir,page,page+'_'+key+'.png'),col_img)

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Page Detection')
    parser.add_argument('--imgdir', type=str)
    parser.add_argument('--clsdir', type=str)
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
    Parallel(n_jobs=-1)(map(delayed(main), pages,[args]*len(pages)))
