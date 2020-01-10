#  				种子杯复赛报告

## **参赛队伍**

两只大白

队长： 吕兴宇  华中科技大学电信学院通信1703班

队员： 汪森林  华中科技大学电信学院通信1703班



## 一、运行环境

### 操作系统和语言

- OS： Ubuntu 18.04


- python 3.6.9



### 依赖库

- pytorch
- pandas
- numpy
- tqdm
- datetime
- time
- csv
- os
- sys



### 运行步骤

```python
# 对net1进行训练
cd net1
python train_net1.py
# 生成net1预测的文件
python gen_pred1.py

# 对net2进行训练
cd ../net2
python train_net2.py
# 生成net2预测的文件
python gen_pred2.py

# 合并net1和net2的预测结果
cd ..
python ensemble.py
```



## 二、项目结构

```
data
	SeedCup_final_train.csv
	SeedCup_final_test.csv
net1
	train_net1.py
	evaluation.py
	model.py
	dataLoader.py
	config.py
	gen_pred.py
net2
	train_net2.py
	evaluation.py
	model.py
	dataLoader.py
	config.py
	gen_pred.py
ensemble.py
```



### 三、主要思路

### 数据分析和预处理

数据分析相较于初赛更为细致和深入, 发现数据有以下特点：

- payed_hour(支付时间的小时) 和shipped_time(发货时间)相关性较强, 将其作为新特征.
- preselling_shipped_time(预售时间)中存在许多噪声, 比如一些时间为 '1970'.
- preselling_shipped_time与shipped_time在某些数据上具备强相关性, 比如一些发货时间和预售时间在同一天.
- seller_uid与company_name以及shipped_prov_name和shipped_city_name基本相互对应.
- 通过分析数据之后, 发现uid, plat_form, product_id与将要预测的值相关性较弱, 将这些特征去除. 


- 将数据中的一些噪声去除, 比如数据中存在的'-99', 以及将preselling_shipped_time中的'1970'等无效的值置为空.



### 模型选择

选择使用神经网络来搭建回归模型:

​	net1 选择特征 biz_type, cate1_id, cate2_id,  cate3_id, seller_uid, company_name, rvcr_prov_name, rvcr_city_name, payed_hour 分别对于(signed_time - payed_time)的day和hour 进行预测.

​	net2 选择特征 biz_type, cate1_id, cate2_id,  cate3_id, seller_uid, compny_name, rvcr_prov_name, rvcr_city_name 分别对于(signed_time - preselling_shipped_time)的day和hour进行预测.

1.神经网络构建:

​	使用FC网络, relu为激活函数, 使用batch normalization, 对于非输出层使用dropout.

2.神经网络简略图示:

​	FC_1(预测day):

![img](file:///C:\Users\11408\AppData\Roaming\Tencent\Users\1140873504\QQ\WinTemp\RichOle\SP_S{LODTO1OFNU2Z88HU7B.png) 

​	FC_2(预测hour):

 ![img](file:///C:\Users\11408\AppData\Roaming\Tencent\Users\1140873504\QQ\WinTemp\RichOle\C827]ATD1I~E7HK0U13O`LQ.png) 

3.预测集成

​	通过net1生成的文件和net2生成的文件进行集成, 将测试集中含有有效preselling_shipped_time的数据使用net2进行预测, 而不含有效preselling_shipped_time的数据使用net1进行预测, 将两个生成的数据进行融合.



### 神经网络优化思路

- 进行分步预测, 对于具有有效preselling_shipped_time的数据, (shipped_time - payed_time) 与shipped_prov_name和prov_city_name具有比较强的相关性, 可以先进行shipped_time的预测, 然后再预测从shipped_time到signed_time的时间差.
- 使用embedding后的向量进行堆叠, 形成一张二维向量图, 对该二维向量图进行卷积神经网络的训练, 然后经过FC层进行回归(主要想提取特征向量), 效果基本没提升, 训练时间变长.
- 使用不同的激活函数 tanh, leakyrelu, elu进行训练, 效果与relu接近, 最后还是选择relu.
- 使用不同的初始化方式xavier_normal, kaiming_normal, 选择xavier_normal.



## 四、 结果分析

| rankScore | main method                                                  |
| --------- | :----------------------------------------------------------- |
| 50.5      | baseline + 自定义损失函数                                    |
| 45.6      | 添加 payed_hour特征进行预测                                  |
| 44.6      | 分两部分预测, 一部分使用preselling_shipped_time, 一部分不使用 |
| 40.8      | net2对(signed_time - preselling_shipped_time)预测, 并将两者集成 |