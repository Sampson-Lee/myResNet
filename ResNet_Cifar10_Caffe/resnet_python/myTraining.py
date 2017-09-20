# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 10:47:14 2017

@author: Sampson
"""

import matplotlib.pyplot as plt  
import caffe 
import numpy as np
caffe.set_device(0)  
caffe.set_mode_gpu()

# 使用SGDSolver，即随机梯度下降算法 
# 使用get_solver，由solver决定算法
solver = caffe.SGDSolver('./res_net_solver.prototxt')

# 等价于solver文件中的max_iter，即最大解算次数  
niter = 10000 
# 每隔1000次收集一次数据  
display= 1000  

# 每次测试进行100次解算，10000/100  
test_iter = 100  
# 每500次训练进行一次测试（100次解算）
test_interval =500 

#初始化 
train_loss = np.zeros(int(niter * 1.0 / display))   
test_loss = np.zeros(int(niter * 1.0 / test_interval))  
test_acc = np.zeros(int(niter * 1.0 / test_interval))


# iteration 0，不计入  
solver.step(1)  

# 辅助变量  
_train_loss = 0;_train_acc=0; _test_loss = 0; _test_acc = 0
# 进行解算
for it in range(niter):
    solver.step(1)  #一次完整的迭代，包括前向、反向和更新
    _train_loss += solver.net.blobs['SoftmaxWithLoss1'].data  #SoftmaxWithLoss1为底端的输出，根据prototxt确定
    #训练集误差   
    if it % display == 0:
        # 计算平均train loss
        train_loss[it // display] = _train_loss / display   # // 是地板除，/ 是普通除法
        train_acc[it // display] = _train_acc / display
        _train_loss = 0
        _train_acc = 0
    #测试集误差
    if it % test_interval == 0:
        for test_it in range(test_iter):
            solver.test_nets[0].forward()   #在测试集执行一个前向过程，test_nets[0]即测试网络的意思
            _test_loss += solver.test_nets[0].blobs['SoftmaxWithLoss1'].data
            _accuracy += solver.test_nets[0].blobs['Accuracy1'].data
        # 计算平均test loss  
        test_loss[it // test_interval] = _test_loss / test_iter
        test_acc[it // test_interval] = _test_acc / test_iter
        _test_loss = 0  
        _test_acc = 0
        
# 绘制train loss、test loss和accuracy曲线  
print('\n plot the train loss and test accuracy \n')
#f, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)  #返回figure对象和ax对象
_, ax1 = plt.subplots()  
ax2 = ax1.twinx()  
# train loss -> 绿色
#记得使用的是arange而不是range或者xrange
ax1.plot(display * np.arange(len(train_loss)), train_loss, 'g')
# test loss -> 黄色  
ax1.plot(test_interval * np.arange(len(test_loss)), test_loss, 'y')
# test accuracy -> 红色  
ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
# train accuracy ->  蓝色
ax2.plot(display * np.arange(len(train_acc)), train_acc, 'b')
ax1.set_xlabel('iteration')  
ax1.set_ylabel('loss')  
ax2.set_ylabel('accuracy')  
plt.savefig("./1.png")  #先保存再显示
plt.show()