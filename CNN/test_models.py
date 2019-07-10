#usr/bin/env python
# coding: utf-8

import tensorflow as tf
import cv2
import os
import numpy as np
import argparse
import keras
from keras.layers import Dropout, Dense,GlobalAveragePooling2D
from keras import Model

def main(testset,weight_path):
    base_dir = testset
    clean_names = lambda x: [i for i in x if i[0]!='.']

    target_names = os.listdir(base_dir)
    target_names = clean_names(target_names)

    num = 0
    for target_name in target_names:
        num += len(os.listdir(os.path.join(base_dir,target_name)))

    cls = []
    for target_name in target_names:
        cls.append(len(os.listdir(os.path.join(base_dir,target_name))))
    class_number = dict(zip(target_names, cls))

    print(f'There are {num} samples in the dataset. \n For each class, we have {class_number}')

    height,width = 80,800
    X = np.zeros((num, height, width), dtype='float')
    y = np.zeros((num,), dtype='int')

    idx = 0
    for cls_id, target_name in enumerate(target_names):
        for path in os.listdir(os.path.join(base_dir,target_name)):
            img = cv2.imread(os.path.join(base_dir,target_name,path),  cv2.IMREAD_GRAYSCALE)
            X[idx, :, :] = cv2.resize(img, (width, height))
            y[idx] = cls_id
            idx += 1

    #model for testing
    base_model = keras.applications.mobilenet.MobileNet(input_shape=(height, width, 1), alpha=1.0,
                                                        depth_multiplier=1, dropout=1e-3, include_top=True,
                                                        weights=None, classes=7)
    with tf.name_scope("output"):
        x = base_model.get_layer("conv_pw_13_relu").output
        x = GlobalAveragePooling2D(data_format=None)(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(target_names), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(weight_path)

    #testing
    res = model.predict(X[..., None])   #input should have four dimension, here:[N,H,W,C=1]
    cls = res.argmax(axis=1)
    count=0
    for i in range(len(cls)):
        if cls[i]!=y[i]:
            count+=1
    acc=1-count/len(cls)
    print('acc on testing set:',acc)
    #import pdb;pdb.set_trace()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Test CNN model')
    parser.add_argument('--testset', type=str)
    parser.add_argument('--weightpath', type=str)
    parser.add_argument('--GPU_num', type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_num
    main(args.testset,args.weightpath)
