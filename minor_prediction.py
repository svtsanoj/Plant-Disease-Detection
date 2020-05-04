# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:56:53 2019

@author: Sanoj
"""

import cv2
import tensorflow as tf

CATEGORIES = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy"]


def prepare(filepath):
    IMG_SIZE = 175  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    #cv2.imshow('img_array',img_array)
    #cv2.imshow('IMG_SIZE',IMG_SIZE)
    #cv2.imshow('new_array',new_array)
    #cv2.waitKey(0)


    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.
    
model = tf.keras.models.load_model("64x3-CNN.model")
prediction = model.predict([prepare("spot2.JPG")])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
