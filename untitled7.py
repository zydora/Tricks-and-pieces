# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:55:39 2019

@author: 46107
"""

from PIL import Image
import torch
import torch.utils.data
#import torch.utils.data.DataLoader as DataLoader
import numpy as np
 
class MyDataset(torch.utils.data.Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, datatxt, transform=True, target_transform=None): #初始化一些需要传入的参数
        super(MyDataset,self).__init__()
        #b = root
        X = np.load(datatxt)#[60000,1,28,28]
        #X_test = np.load('x_test_noisy.npy')#[10000,1,28,28]
        y_train = np.load('y_train.npy')#[60000,]
        y_test = np.load('y_test.npy')#[10000,]
        imgs = []
        if datatxt == 'x_train_noisy.npy':
            for i in range(np.shape(X)[0]):
                imgs.append((X[i],int(y_train[i]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        elif datatxt == 'x_test_noisy.npy':
            for i in range(np.shape(X)[0]):
                imgs.append((X[i],int(y_test[i]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):#这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index] #fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = fn #按照path读入图片from PIL import Image # 按照路径读取图片

        if self.transform is not None:
            img = self.transform(img) #是否进行transform
        return img,label#return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
 
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
 
#根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
        '''
train_data = untitled7.MyDataset(datatxt = 'x_train_noisy.npy')#, transform=transforms.ToTensor())
test_data = untitled7.MyDataset(datatxt = 'x_test_noisy.npy')#, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32)
'''