# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:23:01 2017

@author: Sampson
"""
import caffe
import numpy as np
import matplotlib.pyplot as plt
deploy_dir='./train.prototxt'    #deploy文件
model_dir='./cifar10_resnet_iter_2500.caffemodel'   #训练好的caffemodel
net = caffe.Net(deploy_dir,model_dir,caffe.TEST)   #加载model和network
#对于每一层，看一下数据值（特征图）的形状，通常具有形式(batch_size, channel_dim, height, width)，使用net.blobs
print("feature:\n")
for layer_name, blob in net.blobs.iteritems():
    print(layer_name + '\t' + str(blob.data.shape))

#看一下参数值（卷积核）的形状，通常具有形式(output_channels, input_channels, filter_height, filter_width)，使用net.params
print("filter:\n")
for layer_name, param in net.params.iteritems():
    print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)) # w 和 b 

#data：(num,height,width,channel)
def vis_square(data):
    data = (data - data.min()) / (data.max() - data.min())  #归一化数据
    n = int(np.ceil(np.sqrt(data.shape[0])))  #展成正方形，开方
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))      #为滤波器之间增加空间
                + ((0, 0),) * (data.ndim - 3))  #最后一个维度不填补
    data = np.pad(data, padding, mode='constant', constant_values=1)    #填充白色边界
    #将数据平铺成图像
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')
    plt.show()

import myApp
filters = net.params['Convolution1'][0].data   #提取某层参数（卷积核）
vis_square(filters.transpose(0, 2, 3, 1))
print("\n")
features = net.blobs['Convolution1'].data[0]   #提取某层数据（特征），若没有前向传播特征图的值为0
vis_square(features)
#feature = net.blobs['Convolution1'].data[0,0]
#plt.imshow(feature)
