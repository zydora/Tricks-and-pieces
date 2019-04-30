# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 13:01:21 2019
@author: 46107
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
import pandas as pd
from PIL import Image
import torch
 
np.random.seed(1337)
 
# MNIST dataset
# MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(np.shape(y_train))
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x = torch.tensor(x_train)
y = torch.tensor(x_test)
#print(x[0])
'''
x_train = x.permute(0,3,1,2)
x_test = y.permute(0,3,1,2)

print(np.shape(x_train))
im = x_train
plt.subplot(2, 4, 1)
img = plt.imshow(im[0][0], cmap='gray', interpolation='none')
plt.subplot(2, 4, 2)
img = plt.imshow(im[1][0], cmap='gray', interpolation='none')
plt.subplot(2, 4, 3)
img = plt.imshow(im[2][0], cmap='gray', interpolation='none')
plt.subplot(2, 4, 4)
img = plt.imshow(im[3][0], cmap='gray', interpolation='none')
'''

# Generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std=0.5
# 添加以0.5和STD＝0.5为中心的正常DIST来产生损坏的MNIST图像。
# normal dist（normal distribution),正态分布
noise = np.random.normal(loc=0, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise
 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

x = torch.tensor(x_train_noisy)
y = torch.tensor(x_test_noisy)
#print(x[0])
x_train_noisy = x.permute(0,3,1,2)
x_test_noisy = y.permute(0,3,1,2)


np.random.seed(0)
np.set_printoptions(precision=3)
a = np.random.rand(4, 4)
threshold, upper, lower = 0.5, 1, 0
x_train_noisy[x_train_noisy>threshold] = upper
x_train_noisy[x_train_noisy<threshold] = lower
x_test_noisy[x_test_noisy>threshold] = upper
x_test_noisy[x_test_noisy<threshold] = lower
#print(x_train_noisy[0])

'''
print(np.shape(x_train_noisy))
im = x_train_noisy
for i in range(np.shape(im)[0]):
    img = plt.imshow(im[i][0], cmap='gray', interpolation='none')
    plt.savefig("%d.jpg"%(i+1))
'''


# save
# 适用于保存任何 matplotlib 画出的图像，相当于一个 screencapture



np.save("x_train_noisy.npy",x_train_noisy)
np.save("x_test_noisy.npy",x_test_noisy)
np.save("y_train.npy",y_train)
np.save("y_test.npy",y_test)
#b =  numpy.loadtxt("filename.txt", delimiter=',')
