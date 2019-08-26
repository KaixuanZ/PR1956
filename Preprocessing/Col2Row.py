import json
import cv2
import numpy as np
from scipy import signal
import os
from joblib import Parallel, delayed
import argparse
import multiprocessing
import sys
sys.path.append('../')
import Rect

# read image and detected bounding box, output the image with bounding box

class WarpedImg(object):
    def __init__(self, warpedImg_b, M, colRect, rowHeight=None):
        self.M = M  # transformation from the original img to this wapred img
        self.warpedImg_b = warpedImg_b  # used for initial row segmention
        self.rowRects = []  # rect of each row images in original img
        self.rowHeight = rowHeight
        self.colRect = colRect
        self.rowHeights = []

    def Seg2Rows(self, threshold=10):
        # threshold is for the initial segmentation of row images
        H, W = self.warpedImg_b.shape
        rowLeftIndex, rowRightIndex = [], []
        fgPerRow = np.sum(self.warpedImg_b[:, int(0.1 * W):-int(0.1 * W)], axis=1) / 255
        fgPerRow = signal.medfilt(fgPerRow, 7)

        for i in range(fgPerRow.shape[0] - 1):
            if fgPerRow[i] > threshold and fgPerRow[i + 1] <= threshold:
                rowRightIndex.append(i)
            elif fgPerRow[i] <= threshold and fgPerRow[i + 1] > threshold:
                rowLeftIndex.append(i)

        self.CheckRowIndex(rowLeftIndex, rowRightIndex)
        if len(rowLeftIndex) * len(rowRightIndex) > 1:
            self.GetRowRects(rowLeftIndex, rowRightIndex)
            self.SetRowHeight()
            self.CombineSmallRowRects()
            self.SetRowHeight()
        else:
            rowLeftIndex, rowRightIndex = [0], [H]
            self.GetRowRects(rowLeftIndex, rowRightIndex)
            self.SetRowHeight()

    def CombineSmallRowRects(self):
        for i in range(len(self.rowHeights) - 1, 0 - 1, -1):
            if self.rowHeights[i] < self.rowHeight * 0.8 and len(self.rowHeights)>=2:  # small row: combine it with the closest row
                if i==0:
                    if Rect.DistOfRects(self.rowRects[0], self.rowRects[1]) < self.rowHeight * 3:
                        self.CombineRowRects(0 , 1)
                    else:
                        self.rowRects.pop(i)
                        self.rowHeights.pop(i)
                elif i==len(self.rowHeights)-1:
                    if Rect.DistOfRects(self.rowRects[i], self.rowRects[i-1]) < self.rowHeight * 3:
                        self.CombineRowRects(i - 1, i)
                    else:
                        self.rowRects.pop(i)
                        self.rowHeights.pop(i)
                elif Rect.DistOfRects(self.rowRects[i],self.rowRects[i-1])< Rect.DistOfRects(self.rowRects[i],self.rowRects[i+1]):
                    # combine with the row above
                    self.CombineRowRects(i-1, i)
                else:  # combine with the row below
                    self.CombineRowRects(i, i+1)

    def CombineRowRects(self,i,j):
        if i!=j:
            #combine rect i and j to i
            self.rowRects[i] = Rect.CombineRects(self.rowRects[i], self.rowRects[j])
            self.rowHeights[i] = GetRowHeight(self.rowRects[i])
            self.rowRects.pop(j)
            self.rowHeights.pop(j)

    def SegLargeRows(self, img_b, thetas=list(range(-4, 5))):
        adpRowHeights=signal.medfilt(self.rowHeights, 7)
        for i in range(len(self.rowRects) - 1, -1, -1):
            if GetRowHeight(self.rowRects[i]) >= 2 * min(max(adpRowHeights[i], 0.6*self.rowHeight),1.6*self.rowHeight):
                rowRect = self.rowRects[i]
                rowNum = []
                for theta in thetas:
                    largeRow = self.SegLargeRow(img_b, rowRect, theta)
                    rowNum.append(len(largeRow.rowRects))
                if max(rowNum) > 1:
                    #find the index that's closest to theta=0
                    index = [i for i, x in enumerate(rowNum) if x == max(rowNum)]
                    dict = {}
                    for ii in index:
                        dict[ii] = abs(ii - (len(thetas) - 1) / 2)
                    theta = thetas[min(dict, key=dict.get)]
                    largeRow = self.SegLargeRow(img_b, rowRect, theta, 1)
                    self.rowRects = self.rowRects[0:i] + largeRow.rowRects + self.rowRects[i + 1:]
                    self.rowHeights = self.rowHeights[0:i] + largeRow.rowHeights + self.rowHeights[i + 1:]


    def SegLargeRow(self, img_b, rowRect, theta, f=0):
        rect = [list(rowRect[0]), list(rowRect[1]), rowRect[2]]
        if rect[2]<-45:
            rect[1][0] += 30
        else:
            rect[1][1] += 30
        rect[2] += theta
        warped_b, M = Rect.CropRect(img_b, rect)
        largeRow = WarpedImg(warped_b, M, self.colRect, self.rowHeight)
        largeRow.Seg2Rows()
        # move the center of rows so that they locate within the col
        if f:
            rot = (rowRect[2] - self.colRect[2]) % 90  # relative rotation of row to col (rowRect[2]=colRect[2]+rot)
            if rot > 45:
                rot -= 90
            rot += theta  # relative rotation of rect to col
            for i in range(len(largeRow.rowRects)):
                rect = largeRow.rowRects[i]
                vec = np.array([rect[0][0] - rowRect[0][0], rect[0][1] - rowRect[0][1]])
                vec = vec * np.tan(np.deg2rad(rot))
                vec = [vec[1], -vec[0]]
                largeRow.rowRects[i] = [[rect[0][0] + vec[0], rect[0][1] + vec[1]], rect[1], rect[2]]
        return largeRow

    def CheckRowIndex(self, rowLeftIndex, rowRightIndex):
        # check if leftIndex and rightIndex have the same number, and if leftIndex<rightIndex
        if rowLeftIndex and rowRightIndex:
            if len(rowLeftIndex) > len(rowRightIndex):
                rowLeftIndex.pop(-1)
            elif len(rowLeftIndex) < len(rowRightIndex):
                rowRightIndex.pop(0)
            elif rowLeftIndex[0] > rowRightIndex[0]:
                rowLeftIndex.pop(-1)
                rowRightIndex.pop(0)

    def GetRowRects(self, rowLeftIndex, rowRightIndex):
        H, W = self.warpedImg_b.shape
        for i in range(len(rowLeftIndex)):
            # four pts of the rect
            box = np.array([[0, rowLeftIndex[i]], [W - 1, rowLeftIndex[i]], [0, rowRightIndex[i]], [W - 1, rowRightIndex[i]]])
            rect = Rect.RectOnSrcImg(box, self.M)
            self.rowRects.append(rect)
            self.rowHeights.append(GetRowHeight(rect))

    def SetRowHeight(self):
        if self.rowHeight is None:
            if len(self.rowHeights)>5:
                self.rowHeight = np.percentile(self.rowHeights, 50)  # adaptive estimation of row width
            else:
                self.rowHeight = 50

    def SaveRowJson(self, colJsonName, outputdir='tmp'):
        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)
            print('creating directory ' + outputdir)
        i=0
        for rowRect in self.rowRects:
            rowJsonName=colJsonName.split('.')[0]+'_'+str(i).zfill(3)+'.json'
            with open(os.path.join(outputdir,rowJsonName), 'w') as outfile:
                json.dump(rowRect, outfile)
                if i%50==0:
                    print('output rowRect to ' + os.path.join(outputdir,rowJsonName))
            i+=1

def GetRowHeight(rect):
    if rect[2]<-45:
        return rect[1][0]
    else:
        return rect[1][1]

def Binarization(img,patchSize=15,threshold=12):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # local binarization
    img_b = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, patchSize, threshold)
    return img_b

def GetImgFilename(jsonfile):
    book, f, n , p ,c = jsonfile.split('.')[0].split('_')
    f = f[0] + str(int(f[1:]))
    return book + '_' + f + '_' + n + '.tif'

def RemoveMinistry(img,colRects,colJsonNames):
    rect=Rect.CombineRects(colRects[0],colRects[1])
    ROI=[list(rect[0]),list(rect[1]),rect[2]]
    if ROI[1][0]>ROI[1][1]: #divide width by 4
        ROI[1][1]/=4
    else:
        ROI[1][0]/=4
    img_b=Binarization(img,patchSize=31,threshold=10)
    warped_b , _ =Rect.CropRect(img_b, ROI)
    warped_b=cv2.medianBlur(warped_b, 3)
    warped_b=cv2.medianBlur(warped_b, 3)
    kernel = np.ones([21, 3], np.uint8)
    warped_b = cv2.morphologyEx(warped_b, cv2.MORPH_CLOSE, kernel)
    #use CCL to detected largest region
    ret, labels = cv2.connectedComponents(warped_b)
    size , index = 0 , -1
    Height = lambda x: max(x) - min(x)
    for i in range(1, ret + 1):  # O(n^3)
        if labels[labels == i].shape[0] > warped_b.shape[0]*3 and Height(np.where(labels == i)[0])>size:  # remove small CCL regions
            size , index = Height(np.where(labels == i)[0]) , i
    HRange, _ = np.where(labels == index)
    if min(HRange) > 0.05 * warped_b.shape[0] and min(HRange) < 0.3 * warped_b.shape[0]:
        print("remove Minsitry for "+colJsonNames[0]+" and "+colJsonNames[1])
        for i in range(2):
            H, theta=min(HRange), colRects[i][2]
            colRect=[list(colRects[i][0]),list(colRects[i][1]),colRects[i][2]]
            if colRect[1][0]>colRect[1][1]:
                colRect[1][0]-=H
            else:
                colRect[1][1]-=H
            if theta < -45:
                theta+=90
            colRect[0]=[colRect[0][0]-H*np.sin(np.deg2rad(theta))/2,colRect[0][1]+H*np.cos(np.deg2rad(theta))/2]
            colRects[i]=colRect
        return True
    return False

def main(coldir,imgdir,outputdir):
    clean_names = lambda x: [i for i in x if i[0] != '.']
    colRectJsons = sorted(clean_names(os.listdir(coldir)))

    imgpath = os.path.join(imgdir,GetImgFilename(colRectJsons[0]))
    img = cv2.imread(imgpath)
    img_b = Binarization(img)

    colRects,colJsonNames = [],[]
    for colRectJson in colRectJsons:
        with open(os.path.join(coldir,colRectJson)) as file:
            colRects.append(json.load(file))
            colJsonNames.append(colRectJson)

    if len(colRects)==5:
        RemoveMinistry(img,colRects,colJsonNames)

    for i in range(len(colRects)):
        # detect verticle lines
        col_b, M = Rect.CropRect(img_b, colRects[i])

        col = WarpedImg(col_b, M, colRects[i])
        col.Seg2Rows()
        col.SegLargeRows(img_b)
        col.SegLargeRows(cv2.medianBlur(img_b, 5))
        col.SaveRowJson(colJsonNames[i],outputdir)

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Page Detection')
    parser.add_argument('--imgdir', type=str)
    parser.add_argument('--coldir', type=str)
    parser.add_argument('--outputdir', type=str)
    args = parser.parse_args()

    #create output file
    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
        print('creating directory ' + args.outputdir)

    clean_names = lambda x: [i for i in x if i[0] != '.']
    coldir = os.listdir(args.coldir)
    #coldir = coldir[130::]
    coldir = sorted(clean_names(coldir))

    outputdir = [os.path.join(args.outputdir, dir) for dir in coldir]
    coldir = [os.path.join(args.coldir, dir) for dir in coldir]
    imgdir = [args.imgdir] * len(coldir)

    #Parallel(n_jobs=1)(map(delayed(main), coldir, imgdir, outputdir))
    Parallel(n_jobs=multiprocessing.cpu_count())(map(delayed(main), coldir, imgdir, outputdir))

