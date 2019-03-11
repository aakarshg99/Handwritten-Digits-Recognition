# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:36:31 2018

@author: aakarshg
"""

import numpy as np
from keras.datasets import mnist
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K

#tensorflow backend
#differentiates between channels,rows,columns and rows,columns,channels
K.set_image_dim_ordering('tf')

#random seed fixed for reproducibility
seed = 7
np.random.seed(seed)

#loading the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#reshaping as [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

#Normalize from 0-255 to 0-1
X_train = X_train/255
X_test = X_test/255

#one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

#creating a function for the model
def create_model():
    model = Sequential()
    model.add(Convolution2D(32, (5, 5), input_shape = (28, 28, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    model.add(Convolution2D(16, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(output_dim = 128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    #compiling the model
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

#building the model
model = create_model()

#fitting the model
model.fit(X_train, y_train, batch_size = 200, epochs = 10, verbose = 1, validation_data = (X_test, y_test))

#final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose = 1)

#percentage CNN error
print("CNN error: %0.2f%%" % (100 - scores[1] * 100))

#saving the model
model.save('digit_recognition_model.h5')