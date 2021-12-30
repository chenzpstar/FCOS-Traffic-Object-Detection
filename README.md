# Traffic Object Detection with FCOS

This project is for personal study and under development, please click 'watch' or 'star' my repo and check back later if you are interested in it.

Time: 2022.1.1 - 2022.1.31

Code: SD6

## 1 任务简述

主要任务：在交通场景下，实现对道路目标（车辆和行人）的实时检测。

完成进度：

- [ ] 数据集 ./dataset

    - [ ] kitti.py

    - [ ] bdd100k.py

    - [ ] transform.py

- [ ] 模型 ./model

    - [ ] backbone.py

    - [ ] neck.py

    - [ ] head.py

    - [ ] fcos.py

    - [ ] 训练目标 target.py

    - [ ] 损失函数 loss.py

    - [ ] 后处理 nms.py

- [ ] 工具 ./tool

    - [ ] 评估 eval.py

    - [ ] 训练 train.py

    - [ ] 测试 test.py

    - [ ] 推理 inference.py


## 2 模型

本项目基于Anchor-Free的FCOS算法构建模型。

论文：https://arxiv.org/pdf/2006.09214.pdf

代码：https://github.com/tianzhi0549/FCOS

### 2.1 网络结构

![FCOS](https://d3i71xaburhd42.cloudfront.net/937a93b59c0a70236d34ac1516c26e7676e38967/3-Figure2-1.png)

### 2.2 训练目标

- 类别概率 $$p^* = 1 \ if \ positive \ else \ 0$$

- 边框距离 $$l^* = (x - x_0^{(i)}) / s, \ t^* = (y - y_0^{(i)}) / s \\
r^* = (x_1^{(i)} - x) / s, \ b^* = (y_1^{(i)} - y) / s$$

- 中心程度 $$o^* = \sqrt{\frac{\min(l^*, r^*)}{\max(l^*, r^*)} \times \frac{\min(t^*, b^*)}{\max(t^*, b^*)}}$$

### 2.3 损失函数

$$
\begin{align}
L(\{p_{x,y}\}, \{t_{x,y}\}, \{c_{x,y}\}) = & L_{cls}(p_{x,y}, p_{x,y}^*) + L_{reg}(t_{x,y}, t_{x,y}^*) + L_{ctr}(o_{x,y}, o_{x,y}^*) \\
= & \lambda_{cls} \frac{-1}{N_{pos}} \sum_{x,y} [\alpha (1 - p_{x,y})^\gamma p_{x,y}^* \log(p_{x,y}) + (1 - \alpha) p_{x,y}^\gamma (1 - p_{x,y}^*) \log(1 - p_{x,y})] \\
+ & \lambda_{reg} \frac{1}{N_{pos}} \sum_{x,y} p_{x,y}^* L_{GIoU}(t_{x,y}, t_{x,y}^*) \\
+ & \lambda_{ctr} \frac{-1}{N_{pos}} \sum_{x,y} p_{x,y}^* [o_{x,y}^* \log(o_{x,y}) + (1 - o_{x,y}^*) \log(1 - o_{x,y})]
\end{align}
$$

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
        - label_2
    - testing
        - image_2
```

统计信息：

- 类别数：8

- 类别名称：Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc

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
            - val
            - test
    - labels
        - det_20
            - det_train.json
            - det_val.json
```

统计信息：

![statistics](https://bair.berkeley.edu/static/blog/bdd/bbox_instance.png)

- 类别数：10

- 类别名称：Bus, Light, Sign, Person, Bike, Truck, Motor, Car, Train, Rider

- 训练集图片数：70k

- 验证集图片数：10k

- 测试集图片数：20k

- 图片分辨率：1280x720

- 图片宽高比：1.78:1

## 4 评价指标

本项目基于准确性和实时性指标评价算法性能。

### 4.1 准确性指标

- 交并比 $$IoU = \frac{|B_{p} \cap B_{gt}|}{|B_{p} \cup B_{gt}|}$$

- 混淆矩阵

    |pred \ label|P|N|
    |:--:|:--:|:--:|
    |<b>P|TP|FP|
    |<b>N|FN|TN|

- 精度 $$Precision = \frac{TP}{TP+FP}$$

- 召回率 $$Recall = \frac{TP}{TP+FN}$$

- 准确率 $$Accuracy = \frac{TP+TN}{TP+FP+TN+FN}$$

- F1分数 $$F1-score = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$

- 平均精度 $$AP = \sum_{r=0}^1 (r_{n+1} - r_n) \cdot Percision(r_{n+1})$$

- 平均精度均值 $$mAP = \frac{1}{N}\sum_{i=1}^N AP_{c_i}$$

### 4.2 实时性指标

- 推理帧率 $\ge 25 \ FPS$

- 推理时间 $\le 40 \ ms$