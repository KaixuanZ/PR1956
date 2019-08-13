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
    def __init__(self, warpedImg_b, M, rowWidth=None):
        self.M = M  # transformation from the original img to this wapred img
        self.warpedImg_b = warpedImg_b  # used for initial row segmention
        self.rowRects = []  # rect of each row images in original img
        self.rowWidth = rowWidth

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
            self.CombineNarrowRows(rowLeftIndex, rowRightIndex)
            self.GetRowRects(rowLeftIndex, rowRightIndex)
        else:
            rowLeftIndex, rowRightIndex = [0], [H]
            self.GetRowRects(rowLeftIndex, rowRightIndex)

    def CombineNarrowRows(self, rowLeftIndex, rowRightIndex):
        if len(rowLeftIndex) > 1:
            RowHeights = []
            for i in range(len(rowLeftIndex)):
                RowHeights.append(rowRightIndex[i] - rowLeftIndex[i])
            if self.rowWidth is None:
                self.rowWidth = np.percentile(RowHeights, 30)  # estimation of row width
            for i in range(len(RowHeights) - 1, 0 - 1, -1):
                if RowHeights[i] < self.rowWidth * 0.7:  # small row: combine it with the closest row
                    if i == 0:
                        rowLeftIndex.pop(1)
                        rowRightIndex.pop(0)
                        RowHeights.pop(0)
                        RowHeights[0] = rowRightIndex[0] - rowLeftIndex[0]
                    elif i == len(RowHeights) - 1:
                        rowLeftIndex.pop(-1)
                        rowRightIndex.pop(-2)
                        RowHeights.pop(-1)
                        RowHeights[-1] = rowRightIndex[-1] - rowLeftIndex[-1]
                    elif rowLeftIndex[i] - rowRightIndex[i - 1] < rowLeftIndex[i + 1] - rowRightIndex[i]:  # combine with the row above
                        rowLeftIndex.pop(i)
                        rowRightIndex.pop(i - 1)
                        RowHeights.pop(i)
                        RowHeights[i - 1] = rowRightIndex[i - 1] - rowLeftIndex[i - 1]
                    else:  # combine with the row below
                        rowLeftIndex.pop(i + 1)
                        rowRightIndex.pop(i)
                        RowHeights.pop(i)
                        RowHeights[i] = rowRightIndex[i] - rowLeftIndex[i]

    def SegWideRows(self, img_b, thetas=list(range(-5, 0)) + list(range(1, 6))):
        for i in range(len(self.rowRects) - 1, -1, -1):
            if self.rowRects[i][1][1] >= 2 * self.rowWidth:
                rowRect = self.rowRects[i]
                rowNum = []
                for theta in thetas:
                    wideRow = self.SegWideRow(img_b, rowRect, theta)
                    rowNum.append(len(wideRow.rowRects))
                if max(rowNum) > 1:
                    theta = thetas[rowNum.index(max(rowNum))]
                    wideRow = self.SegWideRow(img_b, rowRect, theta)
                    self.rowRects = self.rowRects[0:i] + wideRow.rowRects + self.rowRects[i + 1:]

    def SegWideRow(self, img_b, rowRect, theta):
        rect = [list(rowRect[0]), list(rowRect[1]), rowRect[2]]
        rect[1][1] += 40
        rect[2] += theta
        warped_b, M = Rect.CropRect(img_b, rect)

        wideRow = WarpedImg(warped_b, M, self.rowWidth)
        wideRow.Seg2Rows()
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

    def SaveRowJson(self, path='rowRects'):
        i = 0
        for rowRect in self.rowRects:
            with open(path + '/row' + str(i) + '.json', 'w') as outfile:
                json.dump(rowRect, outfile)
            i += 1


def Binraization(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # local binarization
    img_b = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    cv2.imshow('img_b', img_b)
    cv2.waitKey(1)
    import pdb;
    pdb.set_trace()
    return img_b

def main():
    img = cv2.imread('pr1956_f53_3.tif')

    with open('pr1956_f0053_3_0_3.json') as file:
        rect = json.load(file)

    img_b = Binraization(img)
    # detect verticle lines
    warped_b, M = Rect.CropRect(img_b, rect)

    col = WarpedImg(warped_b, M)
    col.Seg2Rows()
    col.SaveRowJson('rowRects')
    col.SegWideRows(img_b)
    col.SaveRowJson('rowRects1')

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
    colfilenames = os.listdir(args.coldir)
    colfilenames = sorted(clean_names(colfilenames))
    #pagefilenames = pagefilenames[50:]  #start processing at last checkpoint
    imgdir = [args.imgdir] * len(colfilenames)
    ROIdir = [args.ROIdir] * len(colfilenames)
    outputdir = [args.outputdir] * len(colfilenames)

    Parallel(n_jobs=multiprocessing.cpu_count())(map(delayed(main), colfilenames,imgdir,ROIdir,outputdir))

