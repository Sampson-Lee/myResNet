# -*- coding: utf-8 -*-
import os

import numpy as np
import sklearn
import sklearn.linear_model
from sklearn.model_selection import train_test_split

import lmdb
import caffe

# cifar10原始数据位置
cifar_python_directory = os.path.abspath("cifar-10-batches-py")
print("Converting...")
#cifar10转换为lmdb的位置
cifar_caffe_directory=os.path.abspath("cifar10_train_lmdb")
#如果已经存在路径则说明已经转换
if not os.path.exists(cifar_caffe_directory):
    #解pickle
    def unpickle(file):
        import cPickle
        fo=open(file,'rb')
        dict=cPickle.load(fo)
        fo.close()
        return dict

    
    def shuffle_data(data,labels):
        data,_,labels,_=train_test_split(
            data,labels,test_size=0.0,random_state=42
        )
        return data,labels

    def load_data(train_batches):
        data=[]
        labels=[]
        #将cifar10的5撮数据合并
        for batch in train_batches:
            d=unpickle(
                os.path.join(cifar_python_directory,batch)
            )
            data.append(d['data'])
            labels.append(np.array(d['labels']))
        data=np.concatenate(data)
        labels=np.concatenate(labels)
        length=len(labels)
        #打乱数据
        data,labels=shuffle_data(data,labels)
        return data.reshape(length,3,32,32),labels      #注意数据的格式(length,3,32,32)

    #载入数据
    X,y=load_data(
        ['data_batch_{}'.format(i) for i in range(1,6)]
    )
    Xt,yt=load_data(['test_batch'])

    #将数据转换为lmdb格式，并存储在cifar_caffe_directory
    env=lmdb.open(cifar_caffe_directory,map_size=50000*1000*5)
    txn=env.begin(write=True)
    count=0
    #根据length逐一将数据转为lmdb格式
    for i in range(X.shape[0]):
        datum=caffe.io.array_to_datum(X[i],y[i])
        str_id='{:08}'.format(count)
        txn.put(str_id,datum.SerializeToString())

        count+=1
        if count%1000==0:
            print('already handled with {} pictures'.format(count))
            txn.commit()
            txn=env.begin(write=True)

    txn.commit()
    env.close()

    #将数据转换为lmdb格式，并存储在cifar10_test_lmdb
    env=lmdb.open('cifar10_test_lmdb',map_size=10000*1000*5)
    txn=env.begin(write=True)
    count=0
    for i in range(Xt.shape[0]):
        datum=caffe.io.array_to_datum(Xt[i],yt[i])
        str_id='{:08}'.format(count)
        txn.put(str_id,datum.SerializeToString())

        count+=1
        if count%1000==0:
            print('already handled with {} pictures'.format(count))
            txn.commit()
            txn=env.begin(write=True)

    txn.commit()
    env.close()

else:
    print("Conversion was already done. Did not convert twice.")