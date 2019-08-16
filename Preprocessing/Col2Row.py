import json
import cv2
import numpy as np
from scipy import signal
import os
from joblib import Parallel, delayed
import argparse
import multiprocessing
import Rect

# read image and detected bounding box, output the image with bounding box

class WarpedImg(object):
    def __init__(self, warpedImg_b, M, colRect, rowWidth=None):
        self.M = M  # transformation from the original img to this wapred img
        self.warpedImg_b = warpedImg_b  # used for initial row segmention
        self.rowRects = []  # rect of each row images in original img
        self.rowWidth = rowWidth
        self.colRect = colRect
        self.rowHeights = []

    def Seg2Rows(self, threshold=10):
        # threshold is for the initial segmentation of row images
        H, W = self.warpedImg_b.shape
        rowLeftIndex, rowRightIndex = [], []
        fgPerRow = np.sum(self.warpedImg_b[:, int(0.1 * W):-int(0.1 * W)], axis=1) / 255
        fgPerRow = signal.medfilt(fgPerRow, 15)

        for i in range(fgPerRow.shape[0] - 1):
            if fgPerRow[i] > threshold and fgPerRow[i + 1] <= threshold:
                rowRightIndex.append(i)
            elif fgPerRow[i] <= threshold and fgPerRow[i + 1] > threshold:
                rowLeftIndex.append(i)

        self.CheckRowIndex(rowLeftIndex, rowRightIndex)
        if len(rowLeftIndex) * len(rowRightIndex) > 1:
            self.GetRowRects(rowLeftIndex, rowRightIndex)
            self.CombineNarrowRowRects()
        else:
            rowLeftIndex, rowRightIndex = [0], [H]
            self.GetRowRects(rowLeftIndex, rowRightIndex)

    def CombineNarrowRowRects(self):
        if self.rowWidth is None:
            self.rowWidth = np.percentile(self.rowHeights, 50)  # estimation of row width
        for i in range(len(self.rowHeights) - 1, 0 - 1, -1):
            if self.rowHeights[i] < self.rowWidth * 0.75 and len(self.rowHeights)>=2:  # small row: combine it with the closest row
                if i==0:
                    self.CombineRowRects(0 , 1)
                elif i==len(self.rowHeights)-1:
                    self.CombineRowRects(i - 1, i)
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

    def SegWideRows(self, img_b, thetas=list(range(-4, 5))):
        adpRowHeights=signal.medfilt(self.rowHeights, 7)
        for i in range(len(self.rowRects) - 1, -1, -1):
            if GetRowHeight(self.rowRects[i]) >= 2 * min(adpRowHeights[i],1.6*self.rowWidth):
                rowRect = self.rowRects[i]
                rowNum = []
                for theta in thetas:
                    wideRow = self.SegWideRow(img_b, rowRect, theta)
                    rowNum.append(len(wideRow.rowRects))
                if max(rowNum) > 1:
                    #find the index that's closest to theta=0
                    index = [i for i, x in enumerate(rowNum) if x == max(rowNum)]
                    dict = {}
                    for ii in index:
                        dict[ii] = abs(ii - (len(thetas) - 1) / 2)
                    theta = thetas[min(dict, key=dict.get)]
                    wideRow = self.SegWideRow(img_b, rowRect, theta, 1)
                    self.rowRects = self.rowRects[0:i] + wideRow.rowRects + self.rowRects[i + 1:]
                    self.rowHeights = self.rowHeights[0:i] + wideRow.rowHeights + self.rowHeights[i + 1:]

    def SegWideRow(self, img_b, rowRect, theta, f=0):
        rect = [list(rowRect[0]), list(rowRect[1]), rowRect[2]]
        if rect[2]<-45:
            rect[1][0] += 30
        else:
            rect[1][1] += 30
        rect[2] += theta
        warped_b, M = Rect.CropRect(img_b, rect)
        wideRow = WarpedImg(warped_b, M, self.colRect, self.rowWidth)
        wideRow.Seg2Rows()
        # move the center of rows so that they locate within the col
        if f:
            rot = (rowRect[2] - self.colRect[2]) % 90  # relative rotation of row to col (rowRect[2]=colRect[2]+rot)
            if rot > 45:
                rot -= 90
            rot += theta  # relative rotation of rect to col
            for i in range(len(wideRow.rowRects)):
                rect = wideRow.rowRects[i]
                vec = np.array([rect[0][0] - rowRect[0][0], rect[0][1] - rowRect[0][1]])
                vec = vec * np.tan(np.deg2rad(rot))
                vec = [vec[1], -vec[0]]
                wideRow.rowRects[i] = [[rect[0][0] + vec[0], rect[0][1] + vec[1]], rect[1], rect[2]]
        return wideRow

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

    def SaveRowJson(self, colJsonName, outputdir='tmp'):
        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)
            print('creating directory ' + outputdir)
        i=0
        for rowRect in self.rowRects:
            rowJsonName=colJsonName.split('.')[0]+'_'+str(i).zfill(3)+'.json'
            with open(os.path.join(outputdir,rowJsonName), 'w') as outfile:
                json.dump(rowRect, outfile)
                if i%10==0:
                    print('output rowRect to ' + os.path.join(outputdir,rowJsonName))
            i+=1

def GetRowHeight(rect):
    if rect[2]<-45:
        return rect[1][0]
    else:
        return rect[1][1]

def Binarization(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # local binarization
    img_b = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    return img_b

def GetImgFilename(jsonfile):
    book, f, n , p ,c = jsonfile.split('.')[0].split('_')
    f = f[0] + str(int(f[1:]))
    return book + '_' + f + '_' + n + '.tif'

def main(coldir,imgdir,outputdir):
    clean_names = lambda x: [i for i in x if i[0] != '.']
    colRectJsons = sorted(clean_names(os.listdir(coldir)))

    imgpath = os.path.join(imgdir,GetImgFilename(colRectJsons[0]))
    img = cv2.imread(imgpath)

    for colRectJson in colRectJsons:
        with open(os.path.join(coldir,colRectJson)) as file:
            colRect = json.load(file)

        img_b = Binarization(img)
        # detect verticle lines
        col_b, M = Rect.CropRect(img_b, colRect)

        col = WarpedImg(col_b, M, colRect)
        col.Seg2Rows()
        col.SegWideRows(img_b)
        col.SegWideRows(cv2.medianBlur(img_b, 5))
        col.SaveRowJson(colRectJson,outputdir)

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
    coldir = coldir[1::50]
    coldir = sorted(clean_names(coldir))

    outputdir = [os.path.join(args.outputdir, dir) for dir in coldir]
    coldir = [os.path.join(args.coldir, dir) for dir in coldir]
    imgdir = [args.imgdir] * len(coldir)

    #Parallel(n_jobs=1)(map(delayed(main), coldir, imgdir, outputdir))
    Parallel(n_jobs=multiprocessing.cpu_count())(map(delayed(main), coldir, imgdir, outputdir))

