# ResNet V1 experiments on cifar10 with caffe

## Structure

  The network structure(the identity shortcuts use type B) is here(we only list the network of 20 depth):
  	<br/>[PlainNet_20](http://ethereon.github.io/netscope/#/gist/18200c298ed00d846cfd511babe70a9b)
    <br/>[ResNet_python_20](http://ethereon.github.io/netscope/#/gist/57f30b382aa1e8f32daa75b3bf85cbe5)
  	<br/>[ResNet_paper_20](http://ethereon.github.io/netscope/#/gist/544993a5985bb87e11443dc1dbcb4881)

## Data

Processing
```
   transform_param {
    mean_file: "../mean_pad.binaryproto"
    crop_size: 32    
    mirror:true
  }
```
Input
```
  data_param {
    source: "../cifar10_pad4_train_lmdb"
    batch_size: 128
    backend: LMDB
  }
```

## Usage
  First, you should make sure that your caffe is correctly installed. You can follow this blog's instructions if you use windows.

  for training
  ```
  caffe train -solver=solver.prototxt -gpu 0
  ```

  for testing 
  ```
  caffe test -model=res20_cifar_train_test.prototxt -weights=ResNet_20.caffemodel -iterations=100 -gpu 0
  ```

  Additionally, you are suggested to use train.sh and test.sh just modifing the path

  ## python script
myVisual.py: visualize filters and feature

myAPP.py: apply resnetmodel to inference picture

myRes.py: generalize the prototxt files of train, test, deploy and solver

myTraining.py: train network and plot its process

## Result
### Result with data augmentation:
model|Repeated|Reference
:---:|:---:|:---:
20 lyaers|91.94%|91.25%
32 layers|92.70%|92.49%
44 layers|93.01%|92.83%
56 layers|93.19%|93.03%
110 layers|93.56%|93.39%

**notice**:'Repeated' means reimplementation results and 'Reference' means result in the paper.**We got even better results than the original paper**

### Compare result(without data augmentation):
model|PlainNet|ResNet
:---:|:---:|:---:
20 lyaers|90.10%|91.74%
32 layers|86.96%|92.23%
44 layers|84.45%|92.67%
56 layers|85.26%|93.09%
110 layers|X|93.27%

key: **data augmentation is important!!!**

## Blog address
resnet_paper:
- [Deep Residual Networks学习（一）](https://zhuanlan.zhihu.com/p/22071346)
- [Deep Residual Networks学习(二)](https://zhuanlan.zhihu.com/p/22365736)

resnet_python: 

- [从零开始搭建 ResNet 之 残差网络结构介绍和数据准备](http://www.cnblogs.com/Charles-Wan/p/6442294.html)
- [从零开始搭建 ResNet 之 网络的搭建（上）](http://www.cnblogs.com/Charles-Wan/p/6535395.html#3764266)
- [从零开始搭建 ResNet 之 网络的搭建（中）](http://www.cnblogs.com/Charles-Wan/p/6660077.html#3764263)

## supplement
Variants.md and Comprehension.md are my explorion in Res-family, there are some reference codes: [caffe-model](https://github.com/GeekLiB/caffe-model)