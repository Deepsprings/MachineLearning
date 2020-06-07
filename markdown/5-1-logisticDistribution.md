# 机器学习之逻辑斯蒂回归（Logistic Regression）

> **前言：** 学习笔记，记录下对于一些问题的记录和理解，复习和加深记忆用，挖坑补坑用。
>
> 参考：*李航 《统计学习方法》*

## 0. 基本内容

* 逻辑斯蒂分布（logistic distribution）

  * 分布函数
    $$
    F(x) = \frac{1}{1+e^{-(x-\mu)/\gamma}}
    $$

  * 密度函数
    $$
    f(x) = \frac{e^{-(x-\mu)/\gamma}}{\gamma(1+e^{-(x-\mu)/\gamma})^2}
    $$

  * 图像示意

    ![](/home/zhangwei/Figure_2.png)

* 逻辑斯蒂回归模型

  * 二分类
    $$
    P(Y=1|x) = \frac{e^{-(wx+b)}}{1+e^{-(wx+b)}} \\
    P(Y=0|x) = \frac{1}{1+e^{-(wx+b)}}
    $$

  * 多分类
    $$
    P(Y=k|x) = \frac{e^{-(w_kx+b)}}{1+\sum\limits_{k=1}^{K-1}e^{-(w_kx+b)}},k=1,2,...,K-1 \\
    P(Y=K|x) = \frac{1}{1+\sum\limits_{k=1}^{K-1}e^{-(w_kx+b)}}
    $$

## 1. 问题与理解

考虑其为线性模型，其与感知机的区别与联系，与最大熵模型的区别与联系？

相比于线性回归被广泛应用的原因？

* 线性回归与逻辑斯蒂**回归**

  逻辑斯蒂回归最早是解决分类问题而非回归问题，这点跟线性回归目的不同。逻辑斯蒂回归更像是对直线回归的一种映射，并且是非线性映射, 将回归问题映射为分类问题，同时将解空间映射到 [0, 1] 范围。

  针对线性模型
  $$
  z=w·x+b
  $$
  映射为分类问题的话，一般可考虑使用阶跃函数（unit-step function）
  $$
  y = sgn(z) = \begin{cases}
  1,& z\geq 0\\
  0,& z<0
  \end{cases}
  $$
  但阶跃函数数学性质不好,不可导也不连续.对数几率函数就很好解决了这一问题.即sigmod函数
  $$
  y = \frac{1}{1+e^{-z}}
  $$
  这是最简便和直观的理解

  (**注**:z的正负号无影响,只是w和b的符号问题;分子的形式同理;w,b可用一个向量w表示).

  但问题是,逻辑斯蒂回归一般认为是带有概率性质的,如果那么理解,显然就没有概率属性了,那么这个概率性质是如何来的,应当如何理解.

  **另外:**线性回归是假设 y 服从高斯分布,逻辑回归是假设 y 服从伯努利分布; 线性回归的损失函数是 平方损失,逻辑回归的损失函数是 对数损失.

* 逻辑斯蒂回归的概率属性

  * 从对数几率上来说

    * 事件几率(odds)定义为事件发生的概率p与不发生的概率1-p的比值
      $$
      odds = \frac{p}{1-p} \\
      log\ odds = logit(p) = log \frac{p}{1-p}
      $$

    * 针对逻辑斯蒂回归
      $$
      logit(p) = z = w·x+b
      $$
      反解得
      $$
      y = p = \frac{e^z}{1+e^z} = \frac{e^{w·x+b}}{1+e^{w·x+b}}
      $$
      也即是逻辑斯蒂回归模型,因此可以看出,逻辑斯蒂回归模型就是关于**对数几率**的线性模型.这也是逻辑斯蒂回归的本质:**广义的线性模型**

    * 似乎跟概率扯上了点关系,但好像有点牵强.**一是因为使用的是几率,跟直截了当的概率在直观表现上有差别;二是因为使用了对数,或者更确切地说使用了sigmod, 感觉莫名其妙地使用了.**以至于使得上述解释有些牵强附会,像是先得到结果,然后反推原因.

  * 从贝叶斯理论上来说

    * 参考 *Pattern Recognition and Machine Learning*
  
* 为什么偏偏选用 sigmod 函数?

  * 考虑作为替代阶跃函数的良好选择,具有良好的数学性质
  * 使用tanh呢(**then?**)
  
* 最小化损失函数等价于最大化对数似然函数

  * 使用**极大似然估计**来估计模型参数

    设:
    $$
    P(Y=1|x) = \hat y, P(Y=0|x) = 1-\hat y
    $$
    似然函数:
    $$
    \prod\limits_{i=1}^N\hat y_i^{y_i}[1-\hat y_i]^{1-y_i}
    $$
    对数似然函数:
    $$
    L(w) = \sum\limits_{i=1}^N[y_i log \hat y_i+(1-y_i) log(1-\hat y_i]
    $$
    优化问题:
    $$
    arg\ \mathop{max}\limits_{w} L(w)
    $$
    
* 使用最小化**损失函数**来训练模型
  
  损失函数使用**交叉熵**:
    $$
    L=-[ylog\hat y+(1-y)log(1-\hat y)]
    $$
  
  >**为什么不使用平方损失?**
    >
    >逻辑回归的平方损失函数不是一个凸函数,不容易求解,得到的解是局部最优解.
    >
    >**为什么可以使用交叉熵(对数损失)作为损失函数?**
    >
    >对数损失是个高阶连续可导的凸函数(海塞矩阵半正定)，易于求解;
    >
    >对于可行性,通过二分类问题 y = {1, 0} ，可以借助图像，很好说明
    >
    >* 当 y = 0
    > $$
    >  L = - log(1-\hat y)
    > $$
    >  图像如下:
    >
    >  ![](/home/zhangwei/Figure_3.png)
    >
    >* 当 y = 1
    > $$
    >  L= -log \hat y
    > $$
    >  图像如下 :
    >
    >  ![](/home/zhangwei/Figure_4.png)
    >
    >* 所以对于最小化损失函数:
    > $$
    >  y=0,\ arg \mathop{min}\limits_w L\ => \hat y \rightarrow 0 = y \\
    >  y=1,\ arg \mathop{min}\limits_w L\ => \hat y \rightarrow 1 = y
    > $$
    >  因此可以通过最小化交叉熵来训练模型.
  
  上面 是针对于单一样本,对于整体损失,进行累加:
    $$
    L=- \sum\limits_{i=1}^N[y_i log \hat y_i+(1-y_i) log(1-\hat y_i)]
    $$
    优化问题:
    $$
    arg\ \mathop{min}\limits_wL
    $$
  
* 最优化求解
  
  * 以上
      $$
      L=-L(w)
      $$
      很明显两个最优化问题是等价的
  
  * 常用求解方法
  
    1. **拟牛顿法**：挖坑
  
    2. **梯度下降法**：挖坑
  
* 带正则化项的逻辑回归

  * 正则化的理解（挖坑）
  
  * L1正则
    $$
    L(w) = -{\sum \limits_{i=1}^N[y_i log\hat y_i + (1-y_i)log(1-\hat y_i)+\frac{\lambda}{2}w^2]}
  $$
  
  * L2正则
    $$
    L(w) = -{\sum \limits_{i=1}^N[y_i log\hat y_i + (1-y_i)log(1-\hat y_i)+\frac{\lambda}{2}|w|]}
    $$

* 关于广义线性模型

  * 广义线性模型是基于**指数分布族**，指数分布族原型:
    $$
    P(y;\eta) = b(y)·e^{\eta^TT(y)-\alpha(\eta)}
    $$
    常见指数分布族：二项分布、泊松分布、正态分布、伽马分布等

  * 构建广义线性模型，首先需要满足三个假设

    * y的条件概率分布 $P(y|x ; \theta)$服从指数分布族
      $$
      y|x;\theta \sim ExpFamily(\eta)
      $$

    * 预测T(y)的期望
      $$
      h_\theta(x) = E[T(y)|x]
      $$

    * $\eta$ 与 x 之间是线性的
      $$
      \eta = \theta^Tx
      $$

  * 对于逻辑回归

    * 分布
      $$
      y \sim B(1, \varphi) \\
      P(y) = \varphi^y(1-\varphi)^{(1-y)} = e^{ylog\frac{\varphi}{1-\varphi}+log(1-\varphi)}
      $$
      因此有，对应项
      $$
      b(y) = 1 \\
      \eta = log\frac{\varphi}{1-\varphi}(即logit)\\
      T(y) = y \\
      \alpha(\eta) = -log(1-\varphi)
      $$

    * 预测期望
      $$
      h_\theta(x) = E[T(y)|x] = \varphi
      $$

    * 根据线性关系$\eta = \theta^T x$
      $$
      h_\theta(x) = \frac{1}{1+e^{\theta^Tx}}
      $$

