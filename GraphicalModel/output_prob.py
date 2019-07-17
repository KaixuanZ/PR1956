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
                                                        depth_multiplier=1, dropout=1e-3, include_top=True,
                                                        weights=None, classes=7)
    with tf.name_scope("output"):
        x = base_model.get_layer("conv_pw_13_relu").output
        x = GlobalAveragePooling2D(data_format=None)(x)
        x = Dropout(0.5)(x)
        predictions = Dense(7, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(weightfile)
    return model

def ZeroPadding(filename):
    #for pr1954
    fname,fformat=filename.split('.')
    fname0,fname1,fname2,fname3,fname4=fname.split('_')
    filename=fname0+'_'+fname1+'_'+fname2+'_'+fname3+'_'+fname4.zfill(3)+'.'+fformat
    return filename

def main(inputpath,outputpath,weightfile):
    height,width = 80,800
    clean_names = lambda x: [i for i in x if i[0] != '.']
    if not os.path.isdir(outputpath):
        os.mkdir(outputpath)
        print('creating directory ' + outputpath)

    model=LoadModel(height,width,weightfile)
    count=0
    outputdir=''
    for dir in clean_names(os.listdir(inputpath)):
    #for dir in ['pr1954_p246_0_1','pr1954_p246_0_2','pr1954_p246_0_3','pr1954_p246_0_4','pr1954_p246_0_5']:
        print("processing image in "+dir)
        if outputdir != dir[:-2]:
            outputdir=dir[:-2]
            if not os.path.isdir(os.path.join(outputpath,outputdir)):
                os.mkdir(os.path.join(outputpath,outputdir))
                print('creating directory ' + os.path.join(outputpath,outputdir))
        for img in clean_names(os.listdir(os.path.join(inputpath,dir))):
            data= cv2.imread(os.path.join(inputpath,dir,img), cv2.IMREAD_GRAYSCALE)
            data= cv2.resize(data, (width, height))
            res = model.predict(data[None,...,None])   #input should have four dimension, here:[N=1,H,W,C=1]
            res = res.tolist()[0]
            res = dict(zip(range(len(res)), res))
            img=ZeroPadding(img)    #necessary for dataset without zeropadding


            with open(os.path.join(outputpath,outputdir,img.split('.')[0]+'.json'), 'w') as outfile:
                json.dump(res, outfile)
                count+=1
            if count%10==0:
                print('writing results to ' + os.path.join(outputpath,outputdir,img.split('.')[0]+'.json'))


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
