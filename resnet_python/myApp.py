# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 20:39:31 2017

@author: Sampson
"""

import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
np.seterr(divide='ignore', invalid='ignore')

def preprocess(image,channel,height,width):
	#定义转换也即是预处理函数
	#caffe中用的图像是BGR空间，但是matplotlib用的是RGB空间；
	#再比如caffe的数值空间是[0,255]但是matplotlib的空间是[0,1]。这些都需要转换过来
	transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})  #设定图片的shape格式(64,1,28,28)
	transformer.set_transpose('data',(2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
	transformer.set_mean('data',mu)    #减去均值
	transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间，load_image为0~1之间,要求输入为2值图片则不用#定义转换也即是预处理函数
	transformer.set_channel_swap('data', (2, 1, 0)) #BGR->RGB

	net.blobs['data'].reshape(50,        # batch size,随意设置
	                          channel,         # 3-channel (BGR) images
	                          height, width)  # image size is 28*28
	net.blobs['data'].data[...] = transformer.preprocess('data',image)      #执行上面设置的图片预处理操作，并将图片载入到blob中

if __name__ == '__main__':
    #加载路径
    deploy_dir='./deploy.prototxt'    #deploy文件
    model_dir='./cifar10_resnet_iter_35000.caffemodel'   #caffemodel
    img_dir='./airplane.jpg'    #待测图片
    mean_dir='../data/mean.npy'
    labeltxt_dir='./labels.txt'  #类别名称文件，将数字标签转换回类别名称
    mu=np.load(mean_dir).mean(1).mean(1)
    #print('mean-subtracted values:',zip('BGR',mu))
    
    #加载model和network
    net = caffe.Net(deploy_dir,model_dir,caffe.TEST)
    image = caffe.io.load_image(img_dir)
    preprocess(image,3,28,28)
    #进行分类
    output = net.forward()
    prob= net.blobs['Softmax1'].data[0].flatten()
    print('predicted class is:', prob.argmax())
    labels = np.loadtxt(labeltxt_dir, str, delimiter='\t')
    print('output label:', labels[prob.argmax()])
    top_inds = prob.argsort()[::-1][:5]  # 反转排序并取最大五项
    print('probabilities and labels:', zip(prob[top_inds], labels[top_inds]))
     #显示图片
    img = img.imread(img_dir)
    plt.imshow(image)
    plt.axis('off')
    plt.title(labels[prob.argmax()])
    plt.show()
