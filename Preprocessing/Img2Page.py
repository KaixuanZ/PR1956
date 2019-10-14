import os
import numpy as np
import cv2
import json
import argparse
from joblib import Parallel, delayed

#input scanned img, output bbox of page

def MahalonobisDistance(x, mean, cov):
    # definition of M-distance, not used in this file
    v = x - mean
    dis = np.dot(v, np.linalg.inv(cov))
    dis = np.dot(dis, v.T)
    return dis ** 0.5


def SegByMahalonobisDistance(matrix, mean, cov, thr):
    '''
    :param matrix:  data
    :param mean:    mean vector of a gaussian distribution
    :param cov:     covariance matrix of a gaussian distribution
    :param thr:     threshold for segmentation
    :return:        mask of segmentation results
    '''
    # fast implementation
    cov_inv = np.linalg.inv(cov)
    mat1 = matrix[:, :, 0] - mean[0]
    mat2 = matrix[:, :, 1] - mean[1]
    dis = np.multiply(mat1, mat1) * cov_inv[0, 0] + np.multiply(mat2, mat2) * cov_inv[1, 1] + np.multiply(mat1, mat2) * \
          cov_inv[0, 1] * 2
    return dis < thr ** 2  # mask for segmentation

def TrainGaussian(file):
    '''
    :param file:    filepath of the training data
    :return:        estimated mean vector and covariance matrix
    '''
    img = cv2.imread(file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)

    data = np.concatenate((h.reshape(1, -1), s.reshape(1, -1)), axis=0)

    # parameters to estimate
    mean = np.mean(data, axis=1)
    cov = np.cov(data)
    return mean, cov

def OutputRect(outputdir,filename,rect,splitPage=False):
    '''
    :param outputdir:   output dir
    :param filename:    input page file name
    :param rect:        detected rect of page
    :param splitPage:   split the detect page into two smaller pages if splitPage=True
    :return:            saved rect(s) of page(s)
    '''
    if splitPage:
        #split the rect to two smaller rects (two pages)
        if rect[2] < -45:
            height, width = int(rect[1][0]), int(rect[1][1])
            theta = np.deg2rad(rect[2] + 90)
            norm = width / 4
            vect = [norm * np.cos(theta), norm * np.sin(theta)]
            rect0 = [[rect[0][0] - vect[0], rect[0][1] - vect[1]], [height, width / 2], rect[2]]
            rect1 = [[rect[0][0] + vect[0], rect[0][1] + vect[1]], [height, width / 2], rect[2]]
        else:
            width, height = int(rect[1][0]), int(rect[1][1])
            theta = np.deg2rad(rect[2])
            norm = width / 4
            vect = [norm * np.cos(theta), norm * np.sin(theta)]
            rect0 = [[rect[0][0] - vect[0], rect[0][1] - vect[1]], [width / 2, height], rect[2]]
            rect1 = [[rect[0][0] + vect[0], rect[0][1] + vect[1]], [width / 2, height], rect[2]]
        with open(os.path.join(outputdir, filename + '_0.json'), 'w') as outfile:
            json.dump(rect0, outfile)
            #print('writing results to ' + os.path.join(outputdir, filename + '_0.json'))
        with open(os.path.join(outputdir, filename + '_1.json'), 'w') as outfile:
            json.dump(rect1, outfile)
            #print('writing results to ' + os.path.join(outputdir, filename + '_1.json'))
    else:
        with open(os.path.join(outputdir, filename), 'w') as outfile:
            json.dump(rect, outfile)
            #print('writing results to ' + os.path.join(outputdir, filename))

def main(filename,args):
    #file = '1.tif'
    #mean, cov = TrainGaussian(file)
    #import pdb;pdb.set_trace()
    mean = np.array([20.76549421, 68.80967093])
    cov = np.array([[2.00308826, -7.05376449],
                    [-7.05376449, 46.9934228]])
    thr = 2.5

    print("processing ",filename)
    #import pdb;pdb.set_trace()
    img = cv2.imread(os.path.join(args.inputdir, filename))

    #downsample, faster processing
    img_downsample = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(img)))
    k = 2 ** 3
    img_rgb = cv2.cvtColor(img_downsample, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    #seg in HS space
    mask_HS = SegByMahalonobisDistance(img_hsv[:, :, 0:2], mean, cov, thr)

    #seg in V space
    mask_V = img_hsv[:,:,2]>190

    #combine them
    mask = mask_HS * mask_V

    ret, labels = cv2.connectedComponents(mask.astype(np.uint8))
    size1,label1=0,0
    size2,label2=0,0
    # find the largest two regions
    for i in range(1,ret):
        if np.sum((labels==i).astype(int))>size1:
            size2,label2=size1,label1
            size1=np.sum((labels==i).astype(int))
            label1=i
        elif np.sum((labels==i).astype(int))>size2:
            size2=np.sum((labels==i).astype(int))
            label2=i
    # fit a rect
    _, cnts, _ = cv2.findContours((labels == label1).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect0 = cv2.minAreaRect(cnts[0] * k)

    filename=filename.split('.')[0]
    # seg pages to page if necessary
    if rect0[1][0]*rect0[1][1]>0.76*img.shape[0]*img.shape[1]:
        #print("split rect")
        OutputRect(args.outputdir,filename,rect0,splitPage=True)
    elif rect0[1][0]*rect0[1][1]>0.38*img.shape[0]*img.shape[1]:
        #page(s) may be detected seperately
        _, cnts1,_ = cv2.findContours((labels==label2).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
        rect1=cv2.minAreaRect(cnts1[0]*k)
        if rect1[1][0]*rect1[1][1]>0.35*img.shape[0]*img.shape[1]:
            if rect0[0][0]<rect1[0][0]:
                OutputRect(args.outputdir,filename + '_0.json',rect0)
                OutputRect(args.outputdir,filename + '_1.json',rect1)
            else:
                OutputRect(args.outputdir,filename + '_0.json',rect1)
                OutputRect(args.outputdir,filename + '_1.json',rect0)
        else:
            OutputRect(args.outputdir,filename + '_0.json',rect0)
            print("\n warning: only one output for "+filename+"\n")
    else:
        print("\n warning: no output for "+filename+"\n")

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Page Detection')
    parser.add_argument('--inputdir', type=str)
    parser.add_argument('--outputdir', type=str)
    args = parser.parse_args()

    #create output file
    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
        print('creating directory ' + args.outputdir)

    clean_names = lambda x: [i for i in x if i[0] != '.']
    filenames = sorted(clean_names(os.listdir(args.inputdir)))
    args=[args]*len(filenames)
    Parallel(n_jobs=-1)(map(delayed(main), filenames,args))