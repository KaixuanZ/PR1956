#usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from skimage import transform
from functools import partial
from tensorflow import keras
import cv2
import os
import numpy as np
import argparse

# Define the Data Augmentation Transformation and Data Generator
affine_matrix1 = np.array([[1,0,0],[0.02,1,-5],[0,0,1]])
affine_matrix2 = np.array([[1,0,0],[0.05,1,-20],[0,0,1]])
affine_matrix3 = np.array([[1,0.1,0],[0,1,0],[0,0,1]])
affine_matrix4 = np.array([[1,0.2,0],[0,1,0],[0,0,1]])
affine_matrix5 = np.array([[1,-0.1,0],[0,1,0],[0,0,1]])
affine_matrix6 = np.array([[1,-0.2,0],[0,1,0],[0,0,1]])
affine_matrix7 = np.array([[1,0,0],[-0.02,1,5],[0,0,1]])
affine_matrix8 = np.array([[1,0,0],[-0.05,1,20],[0,0,1]])
affine_matrix9 = np.array([[1,-0.1,0],[0.02,1,-5],[0,0,1]])
affine_matrix10 = np.array([[1,0.1,0],[-0.02,1,5],[0,0,1]])
affine_matrix11 = np.array([[1,0.2,0],[-0.05,1,10],[0,0,1]])
affine_matrix12 = np.array([[1,-0.2,0],[0.05,1,-20],[0,0,1]])

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=32, shuffle=True):
        'Initialization'
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.list_IDs = list(range(len(X)))
        self.n_channels = X.shape[-1]
        self.n_classes = len(y)
        self.shuffle = shuffle
        self.affine_matrice = [
            affine_matrix1, affine_matrix2, affine_matrix3, affine_matrix4,
            affine_matrix5, affine_matrix6, affine_matrix7, affine_matrix8,
            affine_matrix9, affine_matrix10, affine_matrix11, affine_matrix12
        ]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X_batch = self.X[list_IDs_temp]
        y_batch = self.y[list_IDs_temp]

        # Perform Augmentation
        affine_matrix = self.affine_matrice[index%len(self.affine_matrice)]
        affine_transform = partial(transform.warp, inverse_map =                                    transform.AffineTransform(matrix=affine_matrix))
        affine_indice = np.random.permutation(self.batch_size)[:self.batch_size>>1]
        for idx in affine_indice:
            X_batch[idx] = affine_transform(X_batch[idx])

        return X_batch, y_batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

def main(trainset):
    base_dir = trainset
    clean_names = lambda x: [i for i in x if i[0]!='.']

    target_names = os.listdir(base_dir)
    target_names = clean_names(target_names)

    num,cls = 0,[]
    for target_name in target_names:
        num += len(os.listdir(os.path.join(base_dir,target_name)))
        cls.append(len(os.listdir(os.path.join(base_dir,target_name))))
    class_number = dict(zip(target_names, cls))
    #import pdb;pdb.set_trace()
    print(f'There are {num} samples in the dataset. \n For each class, we have {class_number}')

    height,width = 800,80

    X = np.zeros((num, height, width), dtype='float')
    y = np.zeros((num,), dtype='int')

    idx = 0
    for cls_id, target_name in enumerate(target_names):
        for path in os.listdir(os.path.join(base_dir,target_name)):
            img = cv2.imread(os.path.join(base_dir,target_name,path),  cv2.IMREAD_GRAYSCALE)
            X[idx, :, :] = cv2.resize(img, (width, height))

            y[idx] = cls_id
            idx += 1

    X_train, X_val, y_train, y_val = train_test_split(X[...,None], np.eye(len(target_names))[y], test_size=0.15)

    train_generator = DataGenerator(X_train, y_train, 64)
    val_generator = DataGenerator(X_val, y_val, 64)

    model = keras.applications.mobilenet.MobileNet(input_shape=(height, width, 1), alpha=1.0,
                                           depth_multiplier=1, dropout=1e-3, include_top=True,
                                           weights=None, classes=len(target_names))

    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    mc = keras.callbacks.ModelCheckpoint('../../models/weights{epoch:02d}.h5',
                                             save_weights_only=True, period=1)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.0050, patience=6, mode='auto', cooldown=0, min_lr=0)

    model.fit_generator(train_generator, validation_data=val_generator, epochs=40, callbacks=[reduce_lr,mc])


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Train CNN model')
    parser.add_argument('-f', '--trainset', type=str)
    parser.add_argument('-i', '--GPU_num', type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_num
    main(args.trainset)
'''
# In[ ]:


model.save_weights('test_cnn.h5')


# ## Test model performance


res = model.predict(X[:100, ...,None])


# In[ ]:


k = res.argmax(axis=1)
for i in range(40):
    plt.imshow(X[i,...])
    plt.title(target_names[k[i]])
    plt.show()
'''
