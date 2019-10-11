#usr/bin/env python
# coding: utf-8

import json
import tensorflow as tf
import cv2
import os
import argparse
import keras
from keras.layers import Dropout, Dense,GlobalAveragePooling2D
from keras import Model

def LoadModel(height,width,weightfile):
    # model for testing
    base_model = keras.applications.mobilenet.MobileNet(input_shape=(height, width, 1), alpha=1.0,
                                                        depth_multiplier=1, dropout=1e-2, include_top=True,
                                                        weights=None, classes=7)
    with tf.name_scope("output"):
        x = base_model.get_layer("conv_pw_13_relu").output
        x = GlobalAveragePooling2D(data_format=None)(x)
        x = Dropout(0.5)(x)
        predictions = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(weightfile)
    return model

def ZeroPadding(filename):
    #for pr1954
    fname , _ =filename.split('.')
    _ , _ , _ , col , row = fname.split('_')
    return col,row.zfill(3)

def main(inputpath,outputpath,weightfile):
    height,width = 80,800
    clean_names = lambda x: [i for i in x if i[0] != '.']
    if not os.path.isdir(outputpath):
        os.mkdir(outputpath)
        print('creating directory ' + outputpath)

    model=LoadModel(height,width,weightfile)
    for outputfile in clean_names(os.listdir(inputpath)):
        print("processing image in "+outputfile)
        res={}
        for img in sorted(clean_names(os.listdir(os.path.join(inputpath,outputfile)))):
            data= cv2.imread(os.path.join(inputpath,outputfile,img), cv2.IMREAD_GRAYSCALE)
            data= cv2.resize(data, (width, height))
            prob = model.predict(data[None,...,None])   #input should have four dimension, here:[N=1,H,W,C=1]
            prob = prob.tolist()[0]
            prob = dict(zip(range(len(prob)), prob))
            col,row=ZeroPadding(img)    #necessary for dataset without zeropadding

            if col not in res.keys():
                res[col]={}
            res[col][row]=prob

        with open(os.path.join(outputpath,outputfile+'.json'), 'w') as outfile:
            json.dump(res, outfile)
        print('writing results to ' + os.path.join(outputpath,outputfile+'.json'))


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Output Prediction Results')
    parser.add_argument( '--inputpath', type=str)
    parser.add_argument( '--outputpath', type=str)
    parser.add_argument( '--weightfile', type=str)
    parser.add_argument( '--GPU_num', type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_num
    main(args.inputpath,args.outputpath,args.weightfile)
