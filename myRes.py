# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:04:51 2017

@author: Sampson
"""

import caffe
import numpy as np
from caffe import layers as L, params as P, to_proto
import tools

def conv_BN_scale_relu(split, bottom, numout, kernelsize, stride, pad):
    conv = L.Convolution(bottom, kernel_size = kernelsize, stride = stride,
                         num_output=numout,pad=pad,bias_term=True,
                         weight_filler=dict(type = 'msra'),   #统一使用msra
                         bias_filler = dict(type = 'constant'), 
                         param = [dict(lr_mult = 1, decay_mult = 1), 
                                  dict(lr_mult = 2, decay_mult = 0)])
    if split == 'train':
        # 训练的时候我们对 BN 的参数取滑动平均，设置use_global_stats = False
        #如果你不想改变某一层的参数，只要将这一层对应的lr_mult和decay_mult都设置成0即可。
        BN = L.BatchNorm(conv,batch_norm_param=dict(use_global_stats = False),
                         in_place=True,param=[dict(lr_mult=0,decay_mult=0),
                                              dict(lr_mult = 0, decay_mult = 0), 
                                              dict(lr_mult = 0, decay_mult = 0)])
    else:
        # 测试的时候我们直接是有输入的参数，设置use_global_stats = True
        # BN 的学习率惩罚设置为 0，由 scale 学习
        BN = L.BatchNorm(conv,batch_norm_param=dict(use_global_stats = True), 
                         in_place = True, param = [dict(lr_mult = 0, decay_mult = 0), 
                                                   dict(lr_mult = 0, decay_mult = 0), 
                                                   dict(lr_mult = 0, decay_mult = 0)])
    #caffe中 BN 层并不能学习到 α 和 β 参数，因此要加上 scale 层学习
    scale = L.Scale(BN, scale_param = dict(bias_term = True), in_place = True)  #in-place能节省内存
    relu = L.ReLU(scale, in_place = True)
    return scale,relu      

def ResNet_block(split, bottom, numout, kernelsize, stride, projection_stride, pad):
    # 1 代表不需要 1 X 1 的映射
    if projection_stride == 1:        
        scale0 = bottom
    # 否则经过 1 X 1，stride = 2 的映射
    else:
        scale0, relu0 = conv_BN_scale_relu(split,bottom,numout,1,projection_stride,0)
    
    scale1,relu1=conv_BN_scale_relu(split,bottom,numout,kernelsize,projection_stride,pad)
    scale2,relu2=conv_BN_scale_relu(split,relu1,numout,kernelsize,stride,pad)
    wise=L.Eltwise(scale2, scale0, operation = P.Eltwise.SUM)
    wise_relu = L.ReLU(wise, in_place = True)
    
    return wise_relu
    
def ResNet(split):
    # 写入数据的路径
    train_file = '../data/cifar10_train_lmdb'
    test_file = '../data/cifar10_test_lmdb'
    mean_file = '../data/mean.binaryproto'

    # source: 导入的训练数据路径; 
    # backend: 训练数据的格式; 
    # ntop: 有多少个输出,这里是 2 个,分别是 n.data 和 n.labels,即训练数据和标签数据,
    # 对于 caffe 来说 bottom 是输入,top 是输出
    # mirror: 定义是否水平翻转,这里选是
    
    #如果写的是 deploy.prototxt 文件，不用data层
    if split == 'deploy':
        #conv1     
        conv = L.Convolution(bottom='data', kernel_size = 3, stride = 1,
                            num_output=16, pad=1, bias_term=True,
                            weight_filler=dict(type = 'gaussian'),   
                            bias_filler = dict(type = 'constant'), 
                            param = [dict(lr_mult = 1, decay_mult = 1), 
                                    dict(lr_mult = 2, decay_mult = 0)])
        BN = L.BatchNorm(conv,batch_norm_param=dict(use_global_stats = True), 
                         in_place = True, param = [dict(lr_mult = 0, decay_mult = 0), 
                                                   dict(lr_mult = 0, decay_mult = 0), 
                                                   dict(lr_mult = 0, decay_mult = 0)])
        scale = L.Scale(BN, scale_param = dict(bias_term = True), in_place = True)
        result = L.ReLU(scale, in_place = True)
    # 如果写的是训练和测试网络的 prototext 文件
    else:    
        if split == 'train':
            data, labels = L.Data(source=train_file,backend=P.Data.LMDB,
                                batch_size=100, ntop=2,
                                transform_param=dict(mean_value=[0,0,0],
                                                    crop_size=28,
                                                    mirror=True))     
        elif split == 'test':
            data, labels = L.Data(source = test_file, backend = P.Data.LMDB, 
                                batch_size = 100, ntop = 2, 
                                transform_param = dict(mean_value=[0,0,0], 
                                                        crop_size =28))
        #conv1      
        scale, result = conv_BN_scale_relu(split, data, numout = 16, kernelsize = 3, 
                                            stride = 1, pad = 1)


    # 每个 ConvX_X 都有 3 个Residual Block                                                  
    repeat = 3

    # Conv2_X，输入与输出的数据通道数都是 16， 大小都是 32 x 32，可以直接相加，
    # 设置映射步长为 1
    for ii in range(repeat):       
        projection_stride = 1
        result = ResNet_block(split, result, numout = 16, kernelsize = 3, stride = 1, 
                              projection_stride = projection_stride, pad = 1)

    # Conv3_X
    for ii in range(repeat):
    # 只有在刚开始 conv2_X(32 x 32) 到 conv3_X(16 x 16) 的
    # 数据维度不一样，需要映射到相同维度，卷积映射的 stride 为 2
        if ii == 0:
            projection_stride = 2
        else:           
            projection_stride = 1
            
        result = ResNet_block(split, result, numout = 32, kernelsize = 3, stride = 1, 
                          projection_stride = projection_stride, pad = 1)
    # Conv4_X                          
    for ii in range(repeat):
    # 只有在刚开始 conv3_X(16 x 16) 到 conv4_X(8 x 8) 的
    # 数据维度不一样，需要映射到相同维度，卷积映射的 stride 为 2
        if ii == 0:
            projection_stride = 2
        else:
            projection_stride = 1
            
        result = ResNet_block(split, result, numout = 64, kernelsize = 3, stride = 1, 
                          projection_stride = projection_stride, pad = 1)
        
    pool = L.Pooling(result, pool = P.Pooling.AVE, global_pooling = True)
    IP = L.InnerProduct(pool, num_output = 10, weight_filler = dict(type = 'gaussian'), bias_filler = dict(type = 'constant'))
    
    #如果生成deploy文件只需Softmax层
    if split == 'deploy':
        prob = L.Softmax(IP)
        return to_proto(prob)
            
    acc = L.Accuracy(IP, labels)    
    loss = L.SoftmaxWithLoss(IP, labels)

    return to_proto(acc, loss)
    
# 生成 ResNet 网络的 prototxt 文件
def make_net():
    
    # 创建 train.prototxt 并将 ResNet 函数返回的值写入 train.prototxt
    with open(train_dir, 'w') as f:
        f.write('name:"ResNet"\n')
        f.write(str(ResNet('train')))
        
    # 创建 test.prototxt 并将 ResNet 函数返回的值写入 test.prototxt
    with open(test_dir, 'w') as f:
        f.write('name:"ResNet"\n')
        f.write(str(ResNet('test')))

    # 创建 deploy.prototxt 并将 ResNet 函数返回的值写入 deploy.prototxt
    with open(deploy_dir, 'w') as f:
        f.write('name:"ResNet"\n')
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:28\n')
        f.write('input_dim:28\n')
        f.write(str(ResNet('deploy')))

if __name__ == '__main__':
    
    train_dir = './train.prototxt'
    test_dir = './test.prototxt'
    deploy_dir = './deploy.prototxt'
    solver_dir = './res_net_solver.prototxt'
    # 生成train和test文件
    make_net()
    # 生成solver文件
    solver_prototxt = tools.CaffeSolver()
    solver_prototxt.write(solver_dir)