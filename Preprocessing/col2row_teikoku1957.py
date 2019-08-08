# -*- coding: utf-8 -*-
# ---------------------------------------------
# Name:
# Purpose:
#
# Author: Kaixuan Zhang
#
# Created: 2019-06-13
# Copyright: (c) Kaixuan Zhang 2019-06-13
# ----------------------------------------------

from __future__ import division

import argparse
import subprocess
import os
import gc
import time

import cv2
import numpy as np
from scipy import signal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def find_peaks(img,axis):
    ret,thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)   # binarize
    thr=~thr
    thr=thr/255
    pixel_count=np.sum(thr,1-axis)
    threshold=15    #global threshold
    pixel_count=signal.medfilt(pixel_count,5)
    list_left,list_right=[],[]
    for i in range(pixel_count.shape[0]-1):
        if pixel_count[i]>threshold and pixel_count[i+1]<=threshold:
            list_left.append(i)
        elif pixel_count[i]<=threshold and pixel_count[i+1]>threshold:
            list_right.append(i)
    #remove useless pts
    if list_left[0]>list_right[0]:
        list_right.pop(0)
    if list_left[-1]>list_right[-1]:
        list_left.pop(-1)
    #index for img segmentation
    indexes=[int((list_left[i]+list_right[i])/2+0.5) for i in range(len(list_left))]

    #estimate distance between cols/rows
    dist=[indexes[i+1]-indexes[i] for i in range(len(indexes)-1)]
    dist=int(np.median(dist)+0.5)

    threshold=int(1.5*dist+0.5)
    i=0
    while i<len(indexes)-1:
        if indexes[i+1]-indexes[i]>threshold:
            #import pdb;pdb.set_trace()
            #find the missing peaks
            tmp=pixel_count[indexes[i]+dist-int(dist/4+0.5):indexes[i]+dist+int(dist/4+0.5)]
            tmp=tmp.tolist()
            index=tmp.index(min(tmp))+indexes[i]+dist-int(dist/4+0.5)
            indexes.insert(i+1,index)
        i+=1
    return indexes #return index for segmentation


def seg(img, peaks, axis, output_dir):
    H,W=img.shape[:2]
    if axis==1:
        peaks=peaks[::-1]   #seg from right to left
    i = 0
    for k in range(len(peaks) - 1):
        if axis==1: #seg in horizontal direction, from right to left
            output_img = img[:,max(0,peaks[k+1]-10):min(W,peaks[k]+10)]
        else:
            output_img = img[max(0,peaks[k] - 10):min(H,peaks[k + 1] + 10),:]
        cv2.imwrite(os.path.join(output_dir, str(i).zfill(3)+'.png'), output_img)
        i += 1

def seg_img(input_dir, output_dir, file_name):
    """
        seperate the image on vertical direction (axis=0), or horizontal direction (axis=1)
    """
    output_dir_row = os.path.join(output_dir, 'col', file_name)
    if not os.path.isdir(output_dir_row):
        os.mkdir(output_dir_row)
    img = cv2.imread(input_dir, 0)

    axis = 1    #horizontal direction
    peaks = find_peaks(img,axis)

    seg(img, peaks, axis, output_dir_row)


def process_at_file_level(col, ec2_input, ec2_output):
    print_runtime = lambda i, e: \
        print('Finished Task {:01d}. Total Running Time: {:02d}:{:02d}:{:02d}'.
              format(i, e // 3600, (e % 3600 // 60), e % 60))
    start_time = time.time()

    s3_input_file = 's3://teikoku/1957/seg/row/%s.png' % col
    s3_output_col = 's3://teikoku/1957/seg/col/'
    ec2_input_file = os.path.join(ec2_input, '%s.png' % col)

    print('ec2_input_file', ec2_input_file)

    subprocess.call('aws s3 cp %s %s' % (s3_input_file, ec2_input_file), shell=True)

    if not os.path.isdir(ec2_output):
        os.mkdir(ec2_output)

    seg_img(ec2_input_file, ec2_output, col)

    subprocess.call('aws s3 sync %s %s' % (os.path.join(ec2_output, 'col'), s3_output_col),
                    shell=True)

    subprocess.call('rm -r %s' % os.path.join(ec2_output, 'col', '*'), shell=True)
    subprocess.call('rm -r %s' % os.path.join(ec2_input, '*'), shell=True)

    e = int(time.time() - start_time)
    print_runtime(3, e)

    gc.collect()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Seperate image into rows/cols')
    parser.add_argument('-f', '--file_name', type=str,
                        help='file name')
    parser.add_argument('-i', '--input_folder', type=str,
                        help='directory of the input folder')
    parser.add_argument('-o', '--output_folder', type=str,
                        help='directory of the output folder')
    args = parser.parse_args()

    process_at_file_level(args.file_name, args.input_folder, args.output_folder)