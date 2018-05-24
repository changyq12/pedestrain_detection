# Comparing the differences between Faster RCNN and RPN+BF in pedestrain detection

By zyq&cyq

### Introduction

行人检测具有极其广泛的应用：智能辅助驾驶，智能监控，行人分析以及智能机器人等领域。随着深度学习的性能的优越性，将深度学习的方法应用到行人中以提高检测准确率。本工程分别采用Faster R-CNN和RPN+BF网络，对Caltech数据集进行训练和测试，并比较两者的结果。

This code has been tested on Ubuntu 16.04 with MATLAB 2014b and CUDA 7.5.

### Citing 

#### RPN+BF

    @article{zhang2016faster,
      title={Is Faster R-CNN Doing Well for Pedestrian Detection?},
      author={Zhang, Liliang and Lin, Liang and Liang, Xiaodan and He, Kaiming},
      journal={arXiv preprint arXiv:1607.07032},
      year={2016}
    }
    
#### Faster R-CNN

    @article{ren15fasterrcnn,
    Author = {Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun},
    Title = {{Faster R-CNN}: Towards Real-Time Object Detection with Region Proposal Networks},
    Journal = {arXiv preprint arXiv:1506.01497},
    Year = {2015}
    }

### Requirements

0. `Caffe` build for RPN+BF (see [here](https://github.com/zhangliliang/caffe/tree/RPN_BF))
    - If the mex in 'external/caffe/matlab/caffe_faster_rcnn' could not run under your system, please follow the [instructions](https://github.com/zhangliliang/caffe/tree/RPN_BF) on our Caffe branch to compile and replace the mex.

0. MATLAB

0. GPU: Titan X, K40c, etc.

### How to build and run

0. Download the special caffe vision for this project(see [here](https://github.com/zhangliliang/caffe/tree/RPN_BF)), and follow the readme.md in it to build and run.

0. Download the annotations and videos in [Caltech Pedestrian Dataset](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/) and put them in the three folder (videos|res|annotations) under ./RPN_BF/external/code3.2.1/data-USA and ./faster_rcnn_caltech/external/code3.2.1/data-USA.

0. The ./faster_rcnn_caltech include the code of faster rcnn on caltech datasets, follow the readme.md to make sure it perform well. Start MATLAB from the repo folder, and Run `script_faster_rcnn_caltech.m` to train and test the faster rcnn on Caltech, `script_fast_rcnn_caltech_eval.m` to evaluate the result after train and test.

0. The ./RPN_BF include the code of RPN+BF on caltech datasets, follow the readme.md to make sure it perform well, Start MATLAB from the repo folder, and Run `script_rpn_pedestrian_VGG16_caltech` to train and test the RPN model on Caltech, Run `script_rpn_bf_pedestrian_VGG16_caltech` to train and test the BF model on Caltech (the evaluation result is included in the test).

0. Hopefully it would give the evaluation results.  

### Experiment results

#### Faster RCNN

![image](https://github.com/changyq12/pedestrain_detection/raw/master/screenshots/FRCN/ped2.jpg)
![image](https://github.com/changyq12/pedestrain_detection/raw/master/screenshots/FRCN/ped3.jpg)
![image](https://github.com/changyq12/pedestrain_detection/raw/master/screenshots/FRCN/faster-rcnn-stage2.jpg)

In addition, we have raised the mr to 30% for Faster RCNN on the caltech datasets.

#### RPN+BF

![image](https://github.com/changyq12/pedestrain_detection/raw/master/screenshots/RPNBF/ped2.jpg)
![image](https://github.com/changyq12/pedestrain_detection/raw/master/screenshots/RPNBF/ped3.jpg)
![image](https://github.com/changyq12/pedestrain_detection/raw/master/screenshots/RPNBF/rpn.jpg)
![image](https://github.com/changyq12/pedestrain_detection/raw/master/screenshots/RPNBF/rpn_bf.jpg)




