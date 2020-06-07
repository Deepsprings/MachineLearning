## Turing Machine 图灵机

* 基本思想：用机器模拟人类用纸笔数学运算过程

* 图灵机构造：纸带，读写头，控制指令，状态存储器

* 图灵完备：可计算的问题图灵机都能计算，满足这样要求的逻辑系统、装置或编程语言就叫图灵完备

  文章参考：https://www.cs.virginia.edu/~robins/Turing_Paper_1936.pdf

  视频参考：https://www.bilibili.com/video/BV1gQ4y1K7ur

## P与NP问题

* P问题：能在多项式时间内解决的问题

* NP问题：能在多项式时间验证答案正确与否

* NP-hard：比所有NP问题都难的问题

* NP-complete：是个NP问题，同时所有的NP问题都能够约化到它

* P ?= NP：只需证明一个NPC问题是P问题，一般意义上认为P！= NP

  详细参考：http://www.matrix67.com/blog/archives/105
  
  视频参考：https://www.bilibili.com/video/BV1WW411H7nH

## Jensen Inequality 詹森不等式

对于任意点集${x_i}$，若$\lambda_i > 0$，且$\sum\limits_i \lambda_i =1$，则凸函数 g(x) 满足：
$$
g(\sum\limits_{i=1}^M \lambda_i x_i) \leq \sum\limits_{i=1}^M \lambda_i g(x_i)
$$
若考虑$\lambda_i$为随机变量的概率密度，对于凸函数$g(x)$，则
$$
g(E[X]) \leq E[g(x)]
$$

## Jacobi Matrix 雅克比矩阵

* 导数就是线性空间的线性变换，是微分空间到微分空间的线性映射，是微分映射的坐标表示
  $$
  \left\{
  \begin{array}\\
  dx = \frac{\partial x}{\partial \theta_1}d\theta_1 + \frac{\part x}{\part \theta_2}d\theta_2 \\
  dy = \frac{\partial y}{\partial \theta_1}d\theta_1 + \frac{\part y}{\part \theta_2}d\theta_2
  \end{array}
  \right. \\
  即：\left[\begin{matrix}dx\\dy\end{matrix}\right]
  =\left[\begin{matrix} \frac{\part x}{\part \theta_1}&\frac{\part x}{\part \theta_2}\\\frac{\part y}{\part \theta_1}&\frac{\part y}{\part \theta_2}\end{matrix}\right]\left[\begin{matrix}d\theta_1 \\d\theta_2 \end{matrix}\right]
  $$

* 雅克比矩阵是一阶偏导数以一定方式排列形成的矩阵，其行列式成为雅克比行列式
  $$
  J_F(x_1,...,x_n) = \left[
  \begin{matrix}
  \frac{\partial y_1}{\partial x_1} & ...& \frac{\partial y_1}{\partial x_n} \\
  ...&...&... \\
  \frac{\partial y_m}{\partial x_1} & ...& \frac{\partial y_m}{\partial x_n}
  
  \end{matrix}
  \right]
  $$

* 其意义在于它表现为一个多变量函数的最佳线性逼近。可参考，多元函数下的偏导数。
  $$
  F(x) \approx F(p)+J_F(p) ·(x-p)
  $$

* 雅克比行列式的几何意义为矩阵对应线性变换前后的面积比

## Hessian Matrix 海森矩阵

* 海森矩阵是二阶偏导数以一定方式排列的矩阵
  $$
  H(f)=\left[
  \begin{matrix}
  \frac{\partial^2 f}{\part x_1^2}&...&\frac{\partial^2 f}{\part x_1 x_n}\\
  ...&...&... \\
  \frac{\partial^2 f}{\part x_n x_1}&...&\frac{\partial^2 f}{\part x_n^2}
  \end{matrix}
  \right]
  $$
描述函数的局部曲率，可以借助海森矩阵判定多元函数极值。
  
  
  
* 一些应用

  * 应用在牛顿法中，解决最优化问题：使用海森矩阵的逆作为下降幅度
  * 通过海森矩阵判定凸函数：海森矩阵为半正定矩阵时，该函数为凸函数
  * 应用在图像处理中：对与特定结构进行增强，提取关键点

## 最优化理论

* 仿射函数

  最高次数为1的多项式函数，常数项为零的仿射函数称为线性函数。
  $$
  f(x) = Ax + b \\
  展开： f(x) = a_1x_1+a_2x_2+...+a_nx_n+b
  $$
  从$R^n$到$R^m$的映射
  $$
  x \rightarrow Ax+b
  $$
  为仿射变换。

* 凸函数

  在区间 [a, b] 上定义的函数 f， 对于任意两点都有：
  $$
  f(\frac{x_1+x_2}{2}) \leq \frac{f(x_1)+f(x_2)}{2}
  $$
  满足该条件的即为凸函数。当去掉等号条件时，为严格凸函数。

  **注意：**国内外凹凸性定义不尽相同，有时是相反的。

  **特点：**局部最优即为全局最优；仿射函数也是凸函数，不是严格的凸函数。

  **判定：**实数集上，二阶导数在区间上不大于零，为凸函数；恒小于零，为严格凸函数。

* 最优化问题概述

  * 无约束最优化

  $$
  \mathop{min}\limits_{x \in R^n} f(x)
  $$

  * 有约束最优化

    * 等式约束
      $$
      \mathop{min}\limits_{x \in R^n} f(x) \\
      s.t.\ \ \ \ h_j(x)=0  \ \ \ \ j=1,2,3,...,l
      $$

    * 不等式约束
      $$
      \mathop{min}\limits_{x \in R^n} f(x) \\
      s.t.\ \ \ \ g_i(x) \leq 0  \ \ \ \ i=1,2,3,...,k
      $$

* 约束优化问题

    * 一般形式

    $$
    \mathop{min}\limits_{x \in R^n} f(x) \\
    s.t.\ \ \ \ g_i(x) \leq 0  \ \ \ \ i=1,2,3,...,k \\
    \ \ \ \ \ \ \ \ \ \ h_j(x)=0  \ \ \ \ j=1,2,3,...,l
    $$

    * 凸优化问题

        在一般约束优化问题下，同时满足：

        * f(x), g(x), h(x)在定义域内连续可微

        * f(x), g(x) 为凸函数

        * h(x) 为仿射函数

    * 凸二次规划问题

        在满足凸优化问题下，同时满足：

        * f(x) 为二次型函数

        一般形式：
        $$
        \mathop{min}\limits_{x \in R^n} \frac{1}{2}x^TQx+c^Tx \\
        s.t.\ \ \ \ Wx \leq b  \ \ \ \ i=1,2,3,...,k
        $$
        常用求解方法：

        * 椭球法
        * 内点法
        * 增广拉格朗日法
        * 梯度投影法

    

* 拉格朗日对偶性

    * 原始问题 P     $(x^* \rightarrow p^*)$
        $$
        \mathop{min}\limits_{x \in R^n} f(x) \\
        s.t.\ \ \ \ g_i(x) \leq 0  \ \ \ \ i=1,2,3,...,k \\
        \ \ \ \ \ \ \ \ \ \ h_j(x)=0  \ \ \ \ j=1,2,3,...,l
        $$

    * 拉格朗日函数问题 L(=P)       $(x^* \rightarrow l^*)$
        $$
        L(x,\alpha,\beta)=f(x)+\sum\limits_{i=1}^k \alpha_i g_i(x)+\sum\limits_{j=1}^l \beta_i h_j(x) \\
        \mathop{min}\limits_x \ \mathop{max}\limits_{\alpha,\beta}L(x,\alpha,\beta)
        $$

        >关于原始问题等价于拉格朗日函数问题推导：
        >
        >对 L：
        >$$
        >L'=\mathop{max}\limits_{\alpha,\beta}L(\alpha,\beta)=
        >\left\{
        >\begin{aligned}
        >f(x)，\ \ \ 满足约束条件\\
        >+\infty，\ \  不满足约束条件
        >\end{aligned}
        >\right.
        >\\
        >L = \mathop{min}\limits_x L' \Leftrightarrow \mathop{min}\limits_x f(x) =P
        >$$
        >也即等价于原始问题 

    * 对偶问题 D      $(\alpha^\*,\beta^\* \rightarrow d^*)$
        $$
        \mathop{max}\limits_{\alpha,\beta} \mathop{min}\limits_{x}L(x,\alpha,\beta)
        $$

    * 三个重要定理

        * 若原始问题和对偶问题都有最优值，则
            $$
            d^*\leq p^*
            $$

        * 当满足以下条件时，$d^\* = p^\*$

            * 满足凸优化条件：上述
            * 满足slater条件：存在严格满足约束条件的点 x。也即存在x，对于所有 i 有 g(x) <0

        * 原始问题等价于对偶问题等价于满足KKT条件
            $$
            d^* = p^*   \Leftrightarrow  KKT \\
            $$
            

## 函数空间

* 数学空间
  * 是研究工作的对象和遵循的规则
  * 由元素和结构（元素所满足的规则）组成

* 距离

  设X是一个非空集合，任给一对这个几个的元素x,y，有d(x,y)，满足：
  >
  >1. 非负性
  >
  >$$
  >d(x,y) \geq 0;d(x,y)=0 \Leftrightarrow x=y
  >$$
  >
  >2. 对称性
  >
  >$$
  > d(x,y)=d(y,x) \\
  >$$
  >
  >3. 三角不等式
  >
  >$$
  >  d(x,y)\leq d(x,z)+d(z,y)
  >$$
  >
  则d(x,y)称为两点间距离

  **度量空间**：定义了距离的空间（无法描述一个点的“长度”）

* 范数

  若||x||是$R^*$的范数，满足
  >$$
  >||x||\geq 0;||x||=0 \Leftrightarrow x=0\\ 
  >||\alpha x||=|\alpha|||x|| \\
  >||x+y||\leq ||x||+||y||
  >$$
  
  范数可以定义距离：
  $$
  d(x,y)=||x-y||
  $$
  反之，不一定。
  
  **赋范空间**：定义了范数的空间（在度量空间中加入了零点）
  
* 内积

  设$（x, y）\in R$，满足：

  >1. 对称性
  >
  >2. 对第一变元的线性性
  >
  >3. 正定性

  则称（x, y）为内积

  内积可以定义范数：
  $$
  ||x||^2=(x,x)
  $$
  反之，不一定。

  **内积空间**：定义了内积的空间（在赋范空间加入了角度）

* 线性空间

  又称向量空间。可理解为给元素装配了加法和数乘的非空集合。非空集合的两种运算满足八条运算规律。

* 完备性

  空间在极限运算结果，仍然处在该空间中。

  
  
  **巴拿赫空间**：线性完备赋范空间
  
  **欧几里得空间**：有限维线性内积空间
  
  **希尔伯特空间**：无穷维线性完备内积空间
  
  
  
  
  
  