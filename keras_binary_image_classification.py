from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
import cv2
import numpy as np
import os

# data preparation
# as data I choose clocks and crocodiles dataset

# first we need to prepare data: label all data, split data into train and test data

data_path = "/home/nik/Documents/datasets/clockcrocod/"
im = cv2.imread(data_path + 'clock/2251.png')

#cv2.imshow("Picture of clock", im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
