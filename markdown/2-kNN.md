# 机器学习之k近邻算法(k-nearest neighbor, k-NN)

> **前言： ** 学习笔记，记录下对于一些问题的记录和理解，复习和加深记忆用，挖坑补坑用。
>
> 参考：*李航 《统计学习方法》*

# 0. 基本内容

	k近邻算法(kNN)是一种基本分类和回归方法。其不具有显式的学习过程，而是通过给定的训练集合数据，通过k个最近邻的训练实例通过多数表决的方式，进行预测。


# 1. 问题与理解

* 几种距离
	特征空间中两“点” $x_i = (x_i^{(1)}, x_i^{(2)},..., x_i^{(n)})^T , x_j = (x_j^{(1)}, x_j^{(2)},..., x_j^{(n)})^T$，则两点间的 $L_p$距离定义为：
	$$
	L_p(x_i, x_j) = (\sum_{k=1}^{n} |x_i^{(k)} - x_j^{(k)}|^p)^{\frac{1}{p}}
	$$
  * 当p = 1时，称为曼哈顿距离（Manhattan distance）
  * 当p = 2时，称为欧氏距离（Euclidean distance）
  * 当p = ∞时，取值为各个坐标的最大值

* kd树
  * kNN是个简单易理解的算法，可预想到的难点在于，当进行分类时，如何找到最近的k个近邻。一种方法是逐点扫描，计算距离，然后通过比较找到最近的k个点，但显而易见的是，如果训练数据很多，这种方法的计算量会是巨大的。事实上，有很多数据点是不需要考虑和进行计算的。第二种方法kd树便为解决这些问题而生的。



# 2. 具体实例与算法实现
