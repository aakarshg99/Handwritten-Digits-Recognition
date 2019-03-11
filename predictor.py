# -*- coding: utf-8 -*-
"""
Created on Thu May 24 16:16:20 2018

@author: aakarshg
"""

import numpy as np
import cv2
from keras.models import load_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

model = load_model('digit_recognition_model.h5')

img_width, img_height = (28, 28)

img_gray = cv2.imread('eight_one.png', 0)
img_gray.shape

#to convert rgb into B&W
ret, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

"""def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('three_three.png')
img.shape
img_gray = rgb2gray(img)"""

#verifying by plotting if correctly converted
plt.imshow(img_gray, cmap = plt.get_cmap('gray'))
plt.show()

img_gray = cv2.resize(img_gray, (img_width, img_height))
img_gray.shape

img_arr = np.array(img_gray).reshape(img_width, img_height, 1)
img_arr = np.expand_dims(img_arr, axis = 0)
img_arr.shape
#print (img_arr)

pred = model.predict(img_arr)[0]

val = pred[0]
for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    if (pred[n] >= val):
        val = pred[n]
        result = n

print ('The digit is ' + str(result))