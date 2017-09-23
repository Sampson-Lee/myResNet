# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:42:28 2017

@author: Sampson

rely on tensorflow && Keras
"""
import numpy as np
import threading, os, time

labeldict={'airplane':1,
           'automobile':2,
           'bird':3,
           'cat':4,
           'deer':5,
           'dog':6,
           'frog':7,
           'horse':8,
           'ship':9,
           'truck':10           
           }

def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception as e:
        print(str(e))
        return -2
            
#def threadOPS(path, new_path):
#    """
#    多线程处理事务
#    :param src_path: 资源文件
#    :param des_path: 目的地文件
#    :return:
#    """
#    if os.path.isdir(path):
#        img_names = os.listdir(path)
#    else:
#        img_names = [path]
#    for img_name in img_names:
#        print(img_name)
#        tmp_img_name = os.path.join(path, img_name)
#        if os.path.isdir(tmp_img_name):
#            if makeDir(os.path.join(new_path, img_name)) != -1:
#                threadOPS(tmp_img_name, os.path.join(new_path, img_name))
#            else:
#                print('create new dir failure')
#                return -1
#                # os.removedirs(tmp_img_name)
#        elif tmp_img_name.split('.')[1] != "DS_Store":
#            # 读取文件并进行操作
#            image = myData.openImage(tmp_img_name)
#            threadImage = [0] * 5
#            _index = 0
#            for ops_name in opsList:
#                threadImage[_index] = threading.Thread(target=imageOps,
#                                                       args=(ops_name, image, new_path, img_name,))
#                threadImage[_index].start()
#                _index += 1
#                time.sleep(0.2)

def readlmdb(lmdb_dir):
    import caffe
    import lmdb
    dictlmdb={}
    env = lmdb.open(lmdb_dir, readonly=True)
    with env.begin() as txn:
        raw_datum = txn.get(b'00000000')
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)    
    flat_x = np.fromstring(datum.data, dtype=np.uint8)
    dictlmdb['X'] = flat_x.reshape(datum.channels, datum.height, datum.width)
    dictlmdb['y'] = datum.label    
    return dictlmdb

def savelmdb(dictlmdb,lmdb_dir):
    import caffe
    import lmdb
    X=dictlmdb['X']
    y=dictlmdb['y']
    env=lmdb.open(lmdb_dir,map_size=X.nbytes * 20)
    with env.begin(write=True) as txn:
        count=0
        for i in range(X.shape[0]):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tobytes()
            datum.label = int(y[i])
            str_id = '{:08}'.format(i)
            txn.put(str_id,datum.SerializeToString())
    
            count+=1
            if count%1000==0:
                print('already handled with {} pictures'.format(count))
                txn.commit()
                txn=env.begin(write=True)


def readMat(mat_dir):
    import scipy.io as sio
    dictmat = sio.loadmat(mat_dir)
    # 通常需要检查一下data结构，type(data),data.keys(),data.values(),data.items()
    return dictmat

def saveMat(dictmat,mat_dir):
    import scipy.io as sio
    sio.savemat(mat_dir,dictmat)
    print('save mat')
    
def readPickle(pickle_dir):
    import cPickle
    fo=open(pickle_dir,'rb')
    dictpickle=cPickle.load(fo)
    fo.close()
    return dictpickle

def savePickle(dictpickle,pickle_dir):
    import cPickle
    fo=open(pickle_dir,'wb')
    cPickle.dump(fo)
    fo.close()
    print('save pickle')
                 
def readImg(img_dir,label=0,img_size=(128,128)):
    if not img_dir.endswith('/'):
        img_dir += '/'
    
    import cv2        
    img_list = os.listdir(img_dir)
    img_num = len(img_list)
    imgarr = np.zeros((img_num,)+img_size + (3,),dtype=np.int8)
    labelarr = np.ones((img_num,1),dtype=np.int8)*label       
    for i in range(img_num):
        img_name = img_list[i]
        print(img_name)
        print(i)
        img = cv2.cvtColor(cv2.imread(img_dir + img_name), cv2.COLOR_BGR2RGB)  # (height, width, channels)        
        imgarr[i,:,:,:] = cv2.resize(img,img_size,interpolation=cv2.INTER_CUBIC)  # (samples, height, width, channels)
    
    dictimg={}    
    dictimg['X']=imgarr
    dictimg['y']=labelarr
    return dictimg

def augmentation(ndarr_x,ndarr_y=None,epoches=1,save_dir='None',prefix='name',imgformat='jpg'):
    '''
    docs: https://keras-cn.readthedocs.io/en/latest/preprocessing/image/
    param ndarr_x: data.the shape of ndarr_x is (samples, height, width, channels)
    param ndarr_y: labels.the shape of ndarr_y is (samples,label)
    param epoches: samples*epoches*0.1 is number of new samples
    param save_dir: a directory to save the augmented pictures being generated,just for checking
    param name: prefix to use for filenames of saved pictures 
    '''
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
            featurewise_center=False,   # 输入数据集去中心化（均值为0）
            samplewise_center=False,    # 输入数据的每个样本均值为0
            featurewise_std_normalization=False,    # 除以数据集的标准差以完成标准化
            samplewise_std_normalization=False,     # 每个样本除以其自身的标准差
            zca_whitening=False,        # 对输入数据施加ZCA白化
            rotation_range=0.2,         # 图片随机转动的角度
            width_shift_range=0.2,      # 图片水平偏移的幅度
            height_shift_range=0.2,     # 图片竖直偏移的幅度
            shear_range=0.2,            # 剪切强度（逆时针方向的剪切变换角度）
            zoom_range=0.2,             # 随机缩放的幅度,范围是[1-zoom_range, 1+zoom_range]
            channel_shift_range=0.,     # 随机通道偏移的幅度
            fill_mode='nearest',        # 插值方式，‘constant’，‘nearest’，‘reflect’或‘wrap’之一
            cval=0.,                    # 当fill_mode=constant时，指定要向超出边界的点填充的值
            horizontal_flip=True,       # 随机水平翻转
            vertical_flip=True,         # 随机竖直翻转
            rescale=None)               # 重放缩因子,默认为None.如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
    
    if ndarr_x.shape[0] < 10:
        batch_size = 1
    else:
        batch_size=int(ndarr_x.shape[0] * 0.1)
    print(batch_size)
    batch_x=np.zeros((batch_size*epoches,)+ndarr_x.shape[1:])
    batch_y=np.zeros((batch_size*epoches,)+ndarr_y.shape[1:])
    e=0
    #生成器不能用for i in range()
    for xb,yb in datagen.flow(x=ndarr_x,      #
                              y=ndarr_y,      #
                              batch_size = batch_size,  #单次增强图像的数目
                              shuffle=True,             #
                              save_to_dir=save_dir,     #图像存储的路径
                              save_prefix=prefix,       #图像名字的首部
                              save_format=imgformat):   #图像存储的格式
        batch_x[e*batch_size:(e+1)*batch_size,:,:,:] = xb
        batch_y[e*batch_size:(e+1)*batch_size,:,:,:] = yb
        e += 1
        if(e>=epoches):
            break
    return np.concatenate((ndarr_x,batch_x)),np.concatenate((ndarr_y,batch_y))  #将原始数据与增强数据合并后返回

def classifyPictures(img_path, txt_path):
    if not img_path.endswith('/'):
        img_path += '/'    

    import re #正则表达式模块
    import shutil
    fw = open(txt_path,"w")
    labels_key = labeldict.keys()
    labels_val = labeldict.values()
    img_name = os.listdir(img_path) #列出路径中的所有文件
    for itemname in img_name:
        #正则表达式规则:找以label[:]开头,紧跟字符串或者数字0或无限次,并以jpg或者png结尾的图片文件
        for index,keyname in enumerate(labels_key):           
            res = re.search(r'^' + keyname + '[0-9_]*' + '.(jpg|png)$',itemname)
            #只有当返回结果不为空时，进行生成清单和分类图片
            if res != None:
                #生成清单
                fw.write(res.group(0) + ' ' + str(labels_val[index]) + '\n')
                #分类图片
                if makeDir(img_path + keyname) != -2: 
                    shutil.copy(img_path+itemname, img_path + keyname)  #若移动，改为move
                else:
                    print('create new dir failure')               
        print(itemname)    
        
    print("generate txt successfully")
    fw.close()
    
if __name__ == '__main__':
    images_dir = './dog/dog'
    lmdb_dir = './lmdb'
    txt_path = './dog/label.txt'
    pickle_dir='./cifar-10-batches-py/data_batch_1'
    dictimg=readImg(img_dir=images_dir,label=labeldict['dog']) 
#    print(imgndarr.shape)
#    cv2.imshow('image',imgndarr.reshape((128,128,3)))
#    cv2.waitKey(0)
#    new=augmentation(ndarr_x=imgndarr,ndarr_y=None,epoches=10,save_dir=save_path)
#    print(new.shape)
    mat_path = './1.mat'
#    x,y = readMat(mat_path)
#    res=makeDir(images_dir)
#    txt_path = './labellist.txt'
#    classifyPictures(images_dir, txt_path)
#    dictpickle=readPickle(pickle_dir)
#    if makeDir(lmdb_dir) == 0:
#    savelmdb(dictimg,lmdb_dir)
    dictlmdb=readlmdb(lmdb_dir)
#    saveMat(dictimg,mat_path)
#    else:
#        print("Conversion was already done. Did not convert twice, or data just become bigger")
    #常用函数：type(object),ndarray.shape
#    print(type(img))
#    print(type(mat))
