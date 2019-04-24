# S3FD: Single Shot Scale-invariant Face Detector

Designed By [Shifeng Zhang](http://www.cbsr.ia.ac.cn/users/sfzhang/)


## Contents
- [S3FD: Single Shot Scale-invariant Face Detector](#s3fd-single-shot-scale-invariant-face-detector)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Build](#build)
  - [Preparation](#preparation)
    - [Train](#train)
    - [Eval](#eval)
    - [Reference](#reference)


## Installation
1. Get the [SSD](https://github.com/weiliu89/caffe/tree/ssd) code. We will call the directory that you cloned Caffe into `caffe-sfd`
  ```Shell
  git clone https://github.com/weiliu89/caffe.git caffe-sfd
  cd caffe-sfd
  git checkout ssd
  ```

## Build
1. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it. 
2. I follow this [blog](https://blog.csdn.net/lukaslong/article/details/81390276), and it works on my ubuntu 16.4, 1080ti, cuda 8.0, cudnn5.
  
  ```Shell
  # 1. Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config

  # 2. Rewrite Makefile.config
  gedit Makefile.config
    1.若使用cudnn，取消“# USE_CUDNN := 1” 前的注释即：USE_CUDNN := 1
    2.若使用opencv3.x，取消“# OPENCV_VERSION := 3” 前的注释，即：OPENCV_VERSION := 3
    3.取消“# WITH_PYTHON_LAYER := 1” 前的注释。即 WITH_PYTHON_LAYER := 1 
    4.加入hdf5的目录：
      INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
      LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib 
      修改为： 
      INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
      LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial
    5.这一步我是加了，但是有些人没加。。。
      PYTHON_INCLUDE := /usr/include/python2.7 \
                        /usr/lib/python2.7/dist-packages/numpy/core/include
      修改为：
      PYTHON_INCLUDE := /usr/include/python2.7 \
                        /usr/local/lib/python2.7/dist-packages/numpy/core/include
    6.防止打印烦人的警告，CUDA8版本有点高，不支持compute_20
      CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
             -gencode arch=compute_20,code=sm_21 \
             -gencode arch=compute_30,code=sm_30 \
             -gencode arch=compute_35,code=sm_35 \
             -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_52,code=sm_52 \
             -gencode arch=compute_61,code=sm_61
      修改为：
      CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
             -gencode arch=compute_35,code=sm_35 \
             -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_52,code=sm_52 \
             -gencode arch=compute_61,code=sm_61
 
删除这两行即可：
-gencode arch=compute_20,code=sm_20 \
-gencode arch=compute_20,code=sm_21 \
 
删除这两行即可：
-gencode arch=compute_20,code=sm_20 \
-gencode arch=compute_20,code=sm_21 \
  # 3. Rewrite Makefile
  gedit Makefile
    LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_hl hdf5
    修改为：
    LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial

    NVCCFLAGS +=-ccbin=$(CXX) -Xcompiler-fPIC $(COMMON_FLAGS)
    修改为：
    NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)

    LIBRARIES += boost_thread stdc++后加boost_regex
    修改为:
    LIBRARIES += boost_thread stdc++ boost_regex

  # 4. Add caffe-sfd/python to your ~/.bashrc
  sudo gedit ~/.bashrc
    export PYTHONPATH=~/caffe-sfd/python:$PYTHONPATH
  source ~/.bashrc

  # 5. Make
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  # (Optional)
  make runtest -j8
  ```

## Preparation
1. Download [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6). <br>
   By default, we assume the model is stored in `$CAFFE_ROOT/examples/s3fd/`

2. Create the LMDB file.
  ```Shell
  cd $CAFFE_ROOT
  # Create the trainval.txt, test.txt, and test_name_size.txt in data/FACE/
  ./data/FACE/create_list.sh
  # You can modify the parameters in create_data.sh if needed.
  # It will create lmdb files for trainval and test with encoded original image:
  #   - $HOME/data/faces_database/FACE/lmdb/FACE_trainval_lmdb
  #   - $HOME/data/faces_database/FACE/lmdb/FACE_test_lmdb
  # and make soft links at examples/VOC0712/
  ./data/FACE/create_data.sh
  ```

### Train
1. Train your model .
  ```
  ./build/tools/caffe train --solver examples/s3fd/solver.prototxt  --gpu 1 --weights examples/s3fd/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
  ```
 
### Eval
1. ROC of FDDB compared with official, as follow:<br>
![data](https://github.com/lippman1125/github_images/blob/master/s3fd_images/roc.png)

2. ROC of FDDB compared with SSH/MTCNN, as follow:<br>
![data](https://github.com/lippman1125/github_images/blob/master/s3fd_images/roc_compares.png)

3. examples<br>
![data](https://github.com/lippman1125/github_images/blob/master/s3fd_images/example1.jpg)<br>
![data](https://github.com/lippman1125/github_images/blob/master/s3fd_images/example2.jpg)

### Reference
1. https://github.com/sfzhang15/SFD