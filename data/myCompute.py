# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:03:33 2017

@author: Sampson
"""

import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import time
lmdb_env=lmdb.open('E:\DataBase\cifar100\cifar100_train_lmdb')
lmdb_txn=lmdb_env.begin()
lmdb_cursor=lmdb_txn.cursor()
datum=caffe_pb2.Datum()

imgnum=50000
meanchannel = np.zeros((3,imgnum))
stdchannel = np.zeros((3,imgnum)) 
N=0
beginTime = time.time()
for key,value in lmdb_cursor:
    datum.ParseFromString(value)
    data=caffe.io.datum_to_array(datum)
    image=data.transpose(1,2,0)
    meanchannel[0][N] += np.mean(image[:,:,0])
    meanchannel[1][N] += np.mean(image[:,:,1])
    meanchannel[2][N] += np.mean(image[:,:,2])
    stdchannel[0][N] += np.std(image[:,:,0])
    stdchannel[1][N] += np.std(image[:,:,1])
    stdchannel[2][N] += np.std(image[:,:,2])
    N+=1
    if N % 1000 == 0:
        elapsed = time.time() - beginTime
        print("Processed {} images in {:.2f} seconds. "
              "{:.2f} images/second.".format(N, elapsed, N / elapsed))
        
lmdb_env.close()
print(np.mean(meanchannel[0,:]))
print(np.mean(meanchannel[1,:]))
print(np.mean(meanchannel[2,:]))
print(np.mean(stdchannel[0,:]))
print(np.mean(stdchannel[1,:]))
print(np.mean(stdchannel[2,:]))

# use three channel value instead of using mean.binaryproto file
#blob = caffe.io.array_to_blobproto(meanimg)
#with open('mean.binaryproto', 'wb') as f:
#    f.write(blob.SerializeToString())

'''
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( 'mean.binaryproto' , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
print arr
'''