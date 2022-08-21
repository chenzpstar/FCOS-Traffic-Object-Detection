# Traffic Object Detection with FCOS

This project is for personal study and under development, please click 'watch' or 'star' my repo and check back later if you are interested in it.

## 1 任务简述

主要任务：在交通场景下，实现对道路目标（车辆和行人）的实时检测。

目录结构：

```txt
- ./configs 配置
    - kitti_config.py
    - bdd100k_config.py

- ./data 数据
    - kitti.py
    - bdd100k.py
    - transform.py 数据变换
    - collate.py 数据打包

- ./models 模型
    - ./backbones 特征提取网络
        - vgg.py
        - resnet.py
        - darknet.py
        - mobilenet.py
        - shufflenet.py
        - efficientnet.py

    - ./necks 特征融合网络
        - fpn.py
        - pan.py
        - bifpn.py

    - ./layers 网络模块
        - conv.py
        - spp.py
        - aspp.py
        - se.py
        - cbam.py

    - head.py 检测网络
    - target.py 训练目标
    - loss.py 损失函数
    - detect.py 检测后处理
    - fcos.py 完整网络

- ./tools 工具
    - train.py 训练
    - test.py 测试
    - eval.py 评估
    - infer.py 推理
    - demo.py 演示
```

## 2 模型

本项目基于Anchor-Free的FCOS算法构建模型。

论文：https://arxiv.org/pdf/2006.09214.pdf

代码：https://github.com/tianzhi0549/FCOS

### 2.1 网络架构

![FCOS](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-23_at_3.34.09_PM_SAg1OBo.png)

### 2.2 训练目标

- 分类目标

    正样本：![](http://latex.codecogs.com/svg.latex?p^*=1)

    负样本：![](http://latex.codecogs.com/svg.latex?p^*=0)

- 回归目标

    ![](http://latex.codecogs.com/svg.latex?l^*=(x-x_0^{(i)})/s,t^*=(y-y_0^{(i)})/s)

    ![](http://latex.codecogs.com/svg.latex?r^*=(x_1^{(i)}-x)/s,b^*=(y_1^{(i)}-y)/s)

- 中心度目标

    ![](http://latex.codecogs.com/svg.latex?o^*=\sqrt{\frac{\min(l^*,r^*)}{\max(l^*,r^*)}\times\frac{\min(t^*,b^*)}{\max(t^*,b^*)}})

### 2.3 损失函数

- 总损失

    ![](http://latex.codecogs.com/svg.latex?L_{total}({p_{x,y}},{t_{x,y}},{c_{x,y}})=\lambda_{cls}L_{cls}(p_{x,y},p_{x,y}^*)+\lambda_{reg}L_{reg}(t_{x,y},t_{x,y}^*)+\lambda_{ctr}L_{ctr}(o_{x,y},o_{x,y}^*))

- 分类损失：Focal Loss

    ![](http://latex.codecogs.com/svg.latex?L_{cls}(p_{x,y},p_{x,y}^*)=-\frac{1}{N_{pos}}\sum_{x,y}[\alpha(1-p_{x,y})^\gamma%20p_{x,y}^*\log(p_{x,y})+(1-\alpha)p_{x,y}^\gamma(1-p_{x,y}^*)\log(1-p_{x,y})])

- 回归损失：IoU Loss

    ![](http://latex.codecogs.com/svg.latex?L_{reg}(t_{x,y},t_{x,y}^*)=\frac{1}{N_{pos}}\sum_{x,y}p_{x,y}^*L_{IoU}(t_{x,y},t_{x,y}^*))

- 中心度损失：BCE Loss

    ![](http://latex.codecogs.com/svg.latex?L_{ctr}(o_{x,y},o_{x,y}^*)=-\frac{1}{N_{pos}}\sum_{x,y}p_{x,y}^*[o_{x,y}^*\log(o_{x,y})+(1-o_{x,y}^*)\log(1-o_{x,y})])

## 3 数据集

本项目基于公开的KITTI和BDD100K数据集训练模型。

### 3.1 KITTI数据集

官网：http://www.cvlibs.net/datasets/kitti/index.php

论文：http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

目录结构：

```txt
- kitti
    - training
        - image_2
            - 000000.png
        - label_2
            - 000000.txt
    - testing
        - image_2
```

统计信息：

- 类别数：8

- 类别名称：Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc

- 场景：City, Residential, Road, Campus, Person

- 训练集图片数：7481

- 测试集图片数：7518

- 图片分辨率：1224x370、1238x374、1242x375、1241x376

- 图片宽高比：3.3:1

### 3.2 BDD100K数据集

官网：https://bdd-data.berkeley.edu/

文档：https://doc.bdd100k.com/

论文：https://arxiv.org/pdf/1805.04687.pdf

目录结构：

```txt
- bdd100k
    - images
        - 100k
            - train
                - 0000f77c-6257be58.jpg
            - val
            - test
    - labels
        - 100k
            - train
                - 0000f77c-6257be58.json
            - val
```

统计信息：

![statistics](https://bair.berkeley.edu/static/blog/bdd/bbox_instance.png)

- 类别数：10

- 类别名称：Bus, Light, Sign, Person, Bike, Truck, Motor, Car, Train, Rider

- 时间：Dawn/Dusk, Daytime, Night

- 天气：Clear, Partly Cloudy, Overcast, Rainy, Snowy, Foggy

- 场景：Residential, Highway, City Street, Parking Lot, Gas Stations, Tunnel

- 训练集图片数：70k (137张缺少标注)

- 验证集图片数：10k

- 测试集图片数：20k

- 图片分辨率：1280x720

- 图片宽高比：1.78:1

## 4 评价指标

本项目基于准确性和实时性指标评价算法性能。

### 4.1 准确性指标

- 交并比

    ![](http://latex.codecogs.com/svg.latex?IoU=\frac{|B_{p}\cap%20B_{gt}|}{|B_{p}\cup%20B_{gt}|})

- 混淆矩阵

    | pred \ label |   P   |   N   |
    | :----------: | :---: | :---: |
    |     <b>P     |  TP   |  FP   |
    |     <b>N     |  FN   |  TN   |

- 精度

    ![](http://latex.codecogs.com/svg.latex?Precision=\frac{TP}{TP+FP})

- 召回率

    ![](http://latex.codecogs.com/svg.latex?Recall=\frac{TP}{TP+FN})

- F1分数

    ![](http://latex.codecogs.com/svg.latex?F_1-score=\frac{2\cdot%20Precision\cdot%20Recall}{Precision+Recall})

- 平均精度

    ![](http://latex.codecogs.com/svg.latex?AP=\sum_{i=0}^{N-2}(r_{i+1}-r_i)\cdot%20\rho_{interp}(r_{i+1}))

- 平均精度均值

    ![](http://latex.codecogs.com/svg.latex?mAP=\frac{1}{K}\sum_{i=1}^K%20AP_{i})

### 4.2 实时性指标

- 帧率

    ![](http://latex.codecogs.com/svg.latex?FPS\ge25)

- 推理时间

    ![](http://latex.codecogs.com/svg.latex?T\le40ms)
