import json
import cv2
import numpy as np
from scipy import signal
import os
from joblib import Parallel, delayed
import argparse
import sys
sys.path.append('../')
import Rect

#input original scanned img and column bbox, output row bbox

class Page(object):
    def __init__(self,imgpath,colfilename):
        self.columns = []
        self.pagefilename = colfilename.split('/')[-1]
        self.img_b = Binarization(cv2.imread(imgpath,0))

    def AddColumn(self,colRect):
        col_b, M = Rect.CropRect(self.img_b, colRect)
        col = Column(col_b, M, colRect)
        self.columns.append(col)

    def ColsToRows(self):
        for col in self.columns:
            col.SegToRows()
            col.SegLargeRows(self.img_b)
            col.SegLargeRows(cv2.medianBlur(self.img_b, 5))

    def SaveColumnRects(self,outputdir):
        pass

    def SaveRowRects(self,outputdir):
        '''
        :param outputdir
        :return: one json files which contains row rects in this page {'0':[],'1':[],...}
        '''
        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)
            print('creating directory ' + outputdir)

        rowJsonName = self.pagefilename.split('.')[0] + '.json'
        res,i={},0
        for col in self.columns:
            res[i]=col.rowRects
            i+=1
        with open(os.path.join(outputdir, rowJsonName), 'w') as outfile:
            json.dump(res, outfile)

        print('output rowRect to ' + os.path.join(outputdir, rowJsonName))


class Column(object):
    def __init__(self, warpedImg_b, M, colRect, rowHeight=None):
        '''
        :param warpedImg_b: cropped from binarized original image
        :param M:           transformation from the original img to this wapred img
        :param colRect:     rect of this warped image in original image
        :param rowHeight:   rowHeight for estimating large row
        '''
        self.M = M  # transformation from the original img to this wapred img
        self.warpedImg_b = warpedImg_b  # used for initial row segmention
        self.rowRects = []      # rect of each row images in original img
        self.rowHeight = rowHeight
        self.colRect = colRect
        self.rowHeights = []    #[rowheight] for all the segmented rows

    def SegToRows(self, threshold=10):
        '''
        :param threshold
        :return: segmented [row_rects]
        '''
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
        '''
        combine very small rows (rowHeight very small) with closest row
        '''
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
        '''
        :param img_b: binarized original image
        :param thetas: [rotation angle], define the search space
        :return: segmented [row_rect]
        seg rows with large rowHeight
        '''
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
        '''
        :param img_b:
        :param rowRect:
        :param theta: rotation angle
        :param f: flag
        :return: segmented [row_rect]
        '''
        rect = [list(rowRect[0]), list(rowRect[1]), rowRect[2]]
        if rect[2]<-45:
            rect[1][0] += 30
        else:
            rect[1][1] += 30
        rect[2] += theta
        warped_b, M = Rect.CropRect(img_b, rect)
        largeRow = Column(warped_b, M, self.colRect, self.rowHeight)
        largeRow.SegToRows()
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
        '''
        :param rowLeftIndex: a list of start index of rows
        :param rowRightIndex: a list of end index of rows
        :return: [row rect] computed from rowLeftIndex, rowRightIndex
        '''
        H, W = self.warpedImg_b.shape
        for i in range(len(rowLeftIndex)):
            # four pts of the rect
            box = np.array([[0, rowLeftIndex[i]], [W - 1, rowLeftIndex[i]], [0, rowRightIndex[i]], [W - 1, rowRightIndex[i]]])
            rect = Rect.RectOnSrcImg(box, self.M)
            self.rowRects.append(rect)
            self.rowHeights.append(GetRowHeight(rect))

    def SetRowHeight(self):
        '''
        :return: self.rowHeight, a threshold
        '''
        if self.rowHeight is None:
            if len(self.rowHeights)>5:
                self.rowHeight = np.percentile(self.rowHeights, 50)  # adaptive estimation of row width
            else:
                self.rowHeight = 50

    def SaveRowRects(self, colJsonName, outputdir='tmp'):
        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)
            print('creating directory ' + outputdir)

        rowJsonName=colJsonName.split('.')[0]+'.json'
        with open(os.path.join(outputdir,rowJsonName), 'w') as outfile:
            json.dump(self.rowRects, outfile)
        print('output rowRect to ' + os.path.join(outputdir,rowJsonName))

def GetRowHeight(rect):
    if rect[2]<-45:
        return rect[1][0]
    else:
        return rect[1][1]

def Binarization(img,patchSize=15,threshold=12):
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # local binarization
    img_b = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, patchSize, threshold)
    return img_b

def main(colRectJson,args):
    print("processing "+colRectJson)
    imgpath = os.path.join(args.imgdir,'_'.join(colRectJson.split('_')[:-1])+'.png')

    page=Page(imgpath,colRectJson)

    with open(os.path.join(args.coldir, colRectJson)) as file:
        colRects = json.load(file)

    for colRect in colRects:
        page.AddColumn(colRect)

    page.ColsToRows()

    page.SaveRowRects(args.outputdir)

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
    colRectJsons = os.listdir(args.coldir)

    colRectJsons = sorted(clean_names(colRectJsons))

    Parallel(n_jobs=-1)(map(delayed(main), colRectJsons,[args]*len(colRectJsons)))

