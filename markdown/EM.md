

[TOC]

# 机器学习之期望最大化算法(Expectation Maximization, EM)

## 1. 基本内容

* 用于含有**隐变量**的概率模型参数的**极大似然估计**
* EM 是个一般方法，而不是某个具体模型
* 知道有哪些分布模型，但不知道每个样本具体属于哪个分布



## 2. 从三硬币模型

### 2.1 问题提出

**A, B, C** 为三枚硬币，正面朝上的概率假设分别为 $\pi,p,q$ ，进行投掷硬币实验：先投掷 **A** 硬币，结果用 x 表示，如果朝上(x=1)，投掷 **B** 硬币，否则(x=0)，投掷 **C** 硬币。最终结果记为 y，正面 y=1, 反面 y=0

假设经过 n 次实验，得到结果 $y_1, y_2, \cdots, y_n$ ，如何从极大似然估计的角度估计参数？

### 2.2 解决方案

流程如下图所示



<img src="/home/zhangwei/workfiles/blog/ML/figure/em1.png" style="zoom:60%;" />



可以得到联合概率分布：

|            |       y=1        |           y=0           |
| :--------: | :--------------: | :---------------------: |
| from **B** |     $\pi p$      |       $\pi (1-p)$       |
| from **C** |    $(1-\pi)q$    |     $(1-\pi) (1-q)$     |
|   total    | $\pi p+(1-\pi)q$ | $\pi(1-p)+(1-\pi)(1-q)$ |

total其实就是每个结果 $y_i(0\ or\ 1)$ 的密度，对于两种结果，可以统一写成：
$$
\pi p^{y_i}(1-p)^{1-y_i}+(1-\pi)q^{y_i}(1-q)^{1-y_i}
$$
那么转化为极大似然估计问题，就是求解
$$
\mathop{max}\limits_{w,p,q} \prod\limits_{i=1}^n[\pi p^{y_i}(1-p)^{1-y_i}+(1-\pi)q^{y_i}(1-q)^{1-y_i}]
$$
或者可以写成
$$
\mathop{max}\limits_{w,p,q}\prod\limits_{i=1}^n [\pi p+(1-\pi)p]^{x_i}[\pi(1-p)+(1-\pi)(1-q)]^{1-x_i}
$$
对其进行求解，可以得到
$$
p(y=1) = \pi p+(1-\pi)q = \frac{k}{n} \\
p(y=0) = \pi(1-p)+(1-\pi)(1-q) = \frac{n-k}{n}
$$
其中，k 为结果中 y=1 的个数，n 为总实验次数。两式其实是等价的，可以写作为一个关系式
$$
\pi p + (1-\pi)q = \frac{k}{n}
$$
其中，$\pi,p,q$ 三个为待估计参数。方程数小于参数个数（静不定），理论上来讲，只要符合该关系式的参数值，就是一个合理估计。

该问题比较简单，可以求出解析解，但对于一般问题来说，极大似然函数是得不到解析解的。下面从迭代的角度来逼近这个极大似然的解。

### 2.3 换个角度

考虑结果的所有情况，获得条件概率：
$$
\begin{aligned}
P(from B | y=1) &= \frac{\pi p}{\pi p + (1-\pi)q} = a \\
P(from C | y=1) &= \frac{(1-\pi)q}{\pi p + (1-\pi)q} = b\\
P(from B | y=0) &= \frac{\pi (1-p)}{\pi(1-p)+(1-\pi)(1-q)} = c \\
P(from C | y=0) &= \frac{(1-\pi)(1-q)}{\pi(1-p)+(1-\pi)(1-q)} = d
\end{aligned}
$$
也就是说，对与一个实验结果 $y_i$，有上面四种情况（严格来说是两种，因为结果$y_i=0,\ 1$ 是知道的），为了清晰起见，针对每种个实验结果 $y_i$ 可以列出下面的表格

|                 | $y_1$ | $y_2$ | $\cdots$ | $y_n$ |
| :-------------: | :---: | :---: | :------: | :---: |
| from B \| y = 1 | $a_1$ | $a_2$ |          | $a_n$ |
| from C \| y = 1 | $b_1$ | $b_2$ |          | $b_n$ |
| from B \| y = 0 | $c_1$ | $c_2$ |          | $c_n$ |
| from C \| y = 0 | $d_1$ | $d_2$ |          | $d_n$ |
|  $y_i\ total$   |   1   |   1   |          |   1   |

注意，事实上每种结果只会对应两行的情况，即

* 如果 $y_i = 1$ ，只会对应有 $a_i,b_i$，此时 $c_i=0,d_i=0$

* 如果 $y_i=0$ ，只会对应有 $c_i,d_i$，此时 $a_i=0,b_i=0$

基于此， $y_i\ total$ = 1，对于所有结果，有
$$
total = n
$$
那么如何根据这个表格来估计参数 $\pi,p,q$ 呢，其实很简单

* $\pi$ 表示来自于 B 的概率，那么就拿所有来自于 $B$ 的情况占总情况比值来重新估计就可以了，即
  $$
  \pi^{(1)} = \frac{\sum\limits_{i=1}^n a_i +\sum\limits_{i=1}^n c_i}{n}
  $$

* p 表示来自于 B 的条件下，结果为 1 的概率，因此可以得到
  $$
  p^{(1)} = \frac{\sum\limits_{i=1}^n a_i}{\sum\limits_{i=1}^n a_i + \sum\limits_{i=1}^n c_i}
  $$

* q 表示来自 C 的条件下，结果为 1 的概率，可以得到
  $$
  q^{(1)} = \frac{\sum\limits_{i=1}^n b_i}{\sum\limits_{i=1}^n b_i + \sum\limits_{i=1}^n d_i}
  $$

这样实际上就完成了一次迭代过程。也就是 EM 算法的一次迭代过程。



## 3. 到高斯混合模型（Gaussian misture model）

高斯混合模型其实是上面的进一步推广。在三硬币模型中，变量取值是具体的离散值，现在就考虑更普遍的，变量是某一分布的情况（高斯分布），且有多个变量。

关于高斯分布的一些基础知识：

高斯分布 (正态分布) 常记作
$$
N(\mu, \sigma^2)
$$
其中，$\mu$ 为数学期望，$\sigma^2$ 为方差。

一般正态分布转化为标准正态分布
$$
X \sim N(\mu,\sigma^2), \ Y=\frac{X-\mu}{\sigma} \sim N(0,1)
$$
一维高斯分布的概率密度函数为
$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma}exp(- \frac{(x-\mu)^2}{2\sigma^2})
$$

### 3.1 模型建立

**单个高斯模型**

极大似然函数
$$
\begin{aligned}
L(\theta | X) &= log P(X|\theta) \\
& =\sum\limits_{i=1}^n log P(x_i|\theta) \\     
& = \sum\limits_{i=1}^n log\ N(\mu_i,\sigma_i^2)
\end{aligned}
$$
其中，$\theta = (\mu,\sigma)$ ，带入进行求解
$$
\left\{
\begin{aligned}
\frac{\partial L}{\partial \mu} &= 0 \Rightarrow \hat\mu = \frac{1}{n} \sum\limits_{i=1}^n x_i \\
\frac{\partial L}{\partial \sigma} &= 0 \Rightarrow \hat\sigma^2 = \frac{1}{n}\sum\limits_{i=1}^n(x_i-\mu)^2
\end{aligned}
\right.
$$
可以看出，$\mu$ 为样本均值， $\sigma$ 为样本方差

**高斯混合模型**

假如一组数据由多个符合不同的高斯模型组合而成，如下图（以两个为例，N1,N2为两个高斯分布，N为其最终合成分布）

![](/home/zhangwei/workfiles/blog/ML/figure/EM1.png)

推而广之，假设最终数据 y 由 k 个符合不同高斯分布的数据组成，即 $N(\mu_1,\sigma_1),N(\mu_2,\sigma_2),\cdots,N(\mu_n,\sigma_k)$ 

其中用 $\theta=(\mu_1,\mu_2,\cdots,\mu_k,\sigma_1,\sigma_2,\cdots,\sigma_k)$ 表示所有参数，如何表示叠加后的模型。

考虑到最终生成的模型需要满足  $\sum\limits_{i=1}^nP(y_i|\theta) = 1$ ，可以对每个高斯模型赋予一个权重 $\alpha$ 累加，即
$$
P(y|\theta) = \sum\limits_{k=1}^K \alpha_k N(\mu_k,\sigma_k) \\
here,\ \ \sum\limits_{k=1}^K \alpha_k = 1
$$

将权重值加入到估计参数值 $\theta$ 中，即有 $\theta = (\mu_1,\mu_2,\cdots,\mu_k,\sigma_1,\sigma_2,\cdots,\sigma_k,\alpha_1,\alpha_2,\cdots,\alpha_{k-1})$ ，这就是高斯混合模型的表达式。

同上，参数估计如果使用极大似然估计的话，有
$$
\begin{aligned}
L(\theta |Y) &= log P(Y|\theta) \\
& = \sum\limits_{i=1}^n log\ P(y_i|\theta) \\
& = \sum\limits_{i=1}^n log\ [\sum\limits_{k=1}^K\alpha_k N(\mu_k,\sigma_k)]
\end{aligned}
$$
极大化该似然函数，显然有些不太现实，求解是及其困难的。

类似于三硬币模型的推广求解，可以使用迭代方法，逼近其极大值。

### 3.2 问题提出

基于以上，EM 算法估计的高斯混合模型的生成如下：

依据概率 $\alpha_k$ 选择第 k 个高斯分布分模型 $N(\mu_k,\sigma_k)$ ，然后依据这个分模型的概率分布生成观测数据 y。通过 N 次实验，可以观测到的数据结果为 $y_1,y_2,\cdots,y_N$ 。

类比与三硬币模型，如何求解参数的估计呢。

### 3.3 解决方案

用 $\theta_k = (\mu_k,\sigma_k)$ 表示第 k 个高斯模型参数，则类似与三硬币模型，得到条件概率分布

已知第 n 个数据的情况下，其来自于第 k 个高斯模型的条件概率（概率密度）表示为
$$
P(from\ N_k| y_n) = \frac{\alpha_k f(y_n|\theta_k)}{\sum\limits_{k=1}^K \alpha_k f{y_n|\theta_k}} = \gamma_{nk}
$$
其中 $f(y_n|\theta_k)$ 表示第 k 个高斯分布的概率密度。

同三硬币模型一样，可以汇总得到如下表格

|                    |                            $y_1$                             |                            $y_2$                             | $\cdots$ |                            $y_N$                             |
| :----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :------: | :----------------------------------------------------------: |
| from $N_1$ | $y_i$ | $\gamma_{11} = \frac{\alpha_1 f(y_1|\theta_1)}{\sum\limits_{k=1}^K\alpha_k f(y_1|\theta_k)}$ | $\gamma_{21} =\frac{\alpha_1 f(y_2|\theta_1)}{\sum\limits_{k=1}^K \alpha_k f(y_2|\theta_k)}$ |          | $\gamma_{N1} =\frac{\alpha_1 f(y_N|\theta_1)}{\sum\limits_{k=1}^K \alpha_k f(y_N|\theta_k)}$ |
| from $N_2$ | $y_i$ | $\gamma_{12} =\frac{\alpha_2 f(y_1|\theta_2)}{\sum\limits_{k=1}^K\alpha_k f(y_1|\theta_k)}$ | $\gamma_{22} =\frac{\alpha_2 f(y_2|\theta_2)}{\sum\limits_{k=1}^K \alpha_k f(y_2|\theta_k)}$ |          | $\gamma_{N2} =\frac{\alpha_2 f(y_N|\theta2)}{\sum\limits_{k=1}^K \alpha_kf(y_N|\theta_k)}$ |
|     $\cdots $      |                                                              |                                                              |          |                                                              |
| from $N_K$ | $y_i$ | $\gamma_{1K} =\frac{\alpha_K f(y_1|\theta_K)}{\sum\limits_{k=1}\alpha_k f(y_1|\theta_k)}$ | $\gamma_{2K} =\frac{\alpha_K f(y_2|\theta_K)}{\sum\limits_{k=1}^K \alpha_kf(y_2|\theta_k)}$ |          | $\gamma_{NK} =\frac{\alpha_K f(y_N|\theta_K)}{\sum\limits_{k=1}^K \alpha_k f(y_N|\theta_k)}$ |
|       total        |                              1                               |                              1                               |          |                              1                               |

然后同理，根据这个表格估计参数

* $\alpha_k$ 表示数据来自于第 k 个高斯分模型的概率，那么就拿所有来自于 $k$ 的情况占总情况比值来重新估计就可以了，即
  $$
  \alpha_k^{(1)} = \frac{\sum\limits_{n=1}^N \gamma_{nk}}{N}= \frac{\sum\limits_{n=1}^N[\frac{\alpha_k f(y_n|\theta_k)}{\sum\limits_{k=1}^K \alpha_k f{y_n|\theta_k}}]}{N} 
  $$

* $\mu_k$ 表示在第 k 个高斯分模型的条件下，数学期望的值。即
  $$
  \mu_k^{(1)} = \frac{\sum\limits_{n=1}^N \gamma_{nk} y_n}{\sum\limits_{n=1}^N \gamma_{nk}}
  $$

* $\sigma_k$  表示第 k 个高斯分模型的条件下，方差的之。即
  $$
  \sigma_k^{(1)} = \frac{\sum\limits_{n=1}^N \gamma_{nk} (y_n-\mu_k)^2}{\sum\limits_{n=1}^N \gamma_{nk}}
  $$
  

这就完成了一次迭代，也就是 EM 算法中的一次迭代过程。

如果想想整个迭代过程，可以知道从最初的混乱随机的初始化模型，演变到最终的多个可以描述整体的高斯混合模型，有种聚类的效果。而聚类分析也是 EM 算法的一个应用的体现。



## 4. EM算法

### 4.1 总结

上面从简单的三硬币模型到复杂点的高斯混合模型，通过一步步对参数极大似然估计的求解，引入了 EM 算法的迭代过程。

事实上，无论是三硬币模型还是更通用的高斯混合模型，对问题求解过程中主要经历了两步，即 EM 算法中的两步。

* 求解因变量的条件概率分布，即在已知结果数据的情况下，其来自于某一模型（隐变量）的概率。对应于 EM 中的 E 步。
* 根据待估计的参数，建立表格，重新估计参数值。对应于 EM 中的 M 步。

 

### 4.2 主要思想

EM 算法就是为求解不能够求解的极大似然函数，提供一种迭代逼近的方法。推导过程从简单，具体可参考书本，推导思路是：

找到 $\theta$ 值，在每次迭代的时候，迭代后的结果比之前的结果要大，也即使得 $L(\theta)-L(\theta^{(i)})>0$ ，那么整个方向就是向着极大值的方向迭代，直至收敛。 

最终问题可转化为转化为另一个比其小的函数的极大值求解，该函数称为 Q函数，且
$$
Q(\theta,\theta^{(i)}) = \sum\limits_{X}P(X|Y,\theta^{(i)})logP(Y,X|\theta)=E_{X|Y,\Theta^t}L(\theta)
$$
其中，$\theta^{(i)}$ 表示迭代 i 步后的模型参数值，$X=(X_1,X_2,\cdots,X_n)$ 表示未观测数据，$Y=(Y_1,Y_2,\cdots,Y_n)$ 表示观测到的数据。而EM算法的一次迭代就是求Q函数和极大化的过程。

从整个推导转化过程可以知道，该算法并不能找到全局最优解，其整个推导过程引入了不等式，其本质只是指定了迭代方向，这个过程是基于初始值，然后向着该方向发展，最终迭代的结果并不一定是（大概率都不是）全局最优值。只是最终找到了一个能够很好描述已有现象的一个模型。

另外一点，似然函数的求解有点类似于机械系统里面的自由度，求解问题的自由度超过了所给的方程式，但基于许多观测事实和实验结果，找到一个可以描述和预测该系统的最优模型。

整体来看EM算法的本质。因为求解上述极大函数，问题就在于隐变量的未知上，一种求解思想就是通过用隐变量的期望值（均值）来代替隐变量，然后带入到 $L(\theta)$ 中，就能够进行极大似然估计了，这也是EM算法的原型。更进一步的改进是，我们推断出隐变量的概率分布，求解对数似然的期望，然后最大化期望值。

其中Q函数就是对数似然的期望，即
$$
Q(\theta)=E_{X|Y,\Theta^{(i)}}L(\theta)
$$

关于 EM 算法收敛性，可以参照书本。

### 4.3 算法步骤

用 $\theta$ 表示参数，X表示隐变量

* 选择参数的初始值 $\theta^{(0)}$ ，开始迭代

* E步：以当前给定数据以及参数 $\theta^{(i)}$ 推断隐变量的分布，计算隐变量的条件概率分布 $P(X|Y，\theta^{(i)})$ 的期望，或者说是通过推断隐变量分布进而计算完全数据的对数似然关于隐变量的期望。这就是 Q 函数
  $$
  \begin{aligned}
  Q(\theta,\theta^{(i)}) &= E_X[logP(Y,X|\theta)|Y,\theta^{(i)}] \\
  \end{aligned}
  $$

* M步：求 Q 函数对 $\theta$ 的极大值，也就是似然函数问题，即
  $$
  \theta^{(i+1)} = \mathop{arg\ max}\limits_\theta Q(\theta|\theta^{(i)})
  $$

* 重复E、M步，直到收敛 （  $\theta{(i)}$ 与 $\theta^{(i+1)}$ 差距很小 ）



## 5. 实例说明

上面给出了，三硬币模型和高斯混合模型通过求解**条件概率**来进行 EM 算法的过程，可参照上述算法步骤理解。

混合高斯模型具有极大普适性。对于在高斯混合模型下，通过计算完全数据的**对数似然函数的期望**，得到Q函数，然后进行最大化的过程。可参考书中。

上面两种思路是等价的。

## 6. 应用

* 聚类分析（next blog）