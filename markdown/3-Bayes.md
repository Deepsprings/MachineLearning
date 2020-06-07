# 机器学习之朴素贝叶斯法(naive Bayes)

> **前言： ** 学习笔记，记录下对于一些问题的记录和理解，复习和加深记忆用，挖坑补坑用。
>
> 内容参考：*李航 《统计学习方法》*

# 0. 基本内容

$$
P(Y=c_k|X=x) = \frac{P(Y=c_k)P(X=x|Y=c_k)}{P(X=x)}
$$





# 1. 问题与理解

* 贝叶斯定理，朴素贝叶斯法，贝叶斯估计，极大似然估计

  * 贝叶斯定理(生成模型)：$P(Y=c_k|X=x) = \frac{P(Y=c_k)P(X=x|Y=c_k)}{P(X=x)}$

  * 朴素贝叶斯法：是一种方法，用于分类。是基于贝叶斯定理同时对条件概率分布作了**特征条件独立假设**。独立假设即：$P(X=x|Y=c_k) = \sum_{i=1}^{n} P(X^{(i)} = x^{(i)}|Y=c_k)$。这一假设大大简化了模型，同时也是朴素贝叶斯法的由来。

  * 贝叶斯估计，极大似然估计：是一种参数估计方法。

* 极大似然估计与贝叶斯估计推导

  对于一个事件Y，可能取值为 ${y_1, y_2, y_3, ..., y_k}$。现对事件Y进行试验，得到一个样本Y'：${y_4, y_1, y_1, ...., y_k}$，在已知样本情况下，估计Y对应取值的概率。(参考实例：事件Y==>掷骰子; 可能取值==>{1,2,3,4,5,6}; 投掷n次，样本==>{2, 3, 6, 2, .....}, 估计P(Y=n) n=1,2,3,4,5,6)。

  若设${y_1, y_2, y_3, ..., y_k}$对应取值概率设为 ${\theta_1, \theta_2, \theta_3, ..., \theta_k}$，即估计 $\theta_i$值。

  * 极大似然估计

    其假定 $\theta_i$ 为定值。

    对于样本${y_4, y_1, y_1, ...., y_k}$，似然函数表示为：
    $$
    l(\theta) = \theta_4 \theta_1 \theta_1 .....\theta_k=\theta_1^{m_1} \theta_2^{m_2} \theta_3^{m_3}...\theta_k^{m_k} \ (m_k表示样本中y_k出现的次数)
    $$
    取对数：
    $$
    ln(l(\theta)) = m_1ln\theta_1+m_2ln\theta_2+...+m_kln\theta_k
    $$
    问题等价于：
    $$
    \left\{
    \begin{aligned}
    求解\theta：\ \ \ \ \ \ \ \underset{\theta}{max}\ ln(l(\theta)) \\
    s.t.\ \ \theta_1+\theta_2+...\theta_k = 1 
    \end{aligned}
    \right.
    $$
    此为有等式约束的最优化问题，使用拉格朗日乘子法，构建拉格朗日函数：
    $$
    L(\theta) = ln(l(\theta)) + \lambda(\theta_1+\theta_2+...+\theta_k-1)
    $$
    求偏导 $\frac{\partial L(\theta)}{\partial \theta_i} = 0$得：
    $$
    \frac{\partial L(\theta)}{\partial \theta_i} = 0\ \ \ \ => \ \ \ \ \left\{
    \begin{array}\\
    \frac{m_1}{\theta_1}+\lambda = 0 &  =>  &\theta_1=-\frac{m_1}{\lambda}\\
    \frac{m_2}{\theta_2}+\lambda = 0 &  =>  &\theta_2=-\frac{m_2}{\lambda} \\
    ...... \\
    \frac{m_k}{\theta_k}+\lambda = 0 &  =>  &\theta_k=-\frac{m_k}{\lambda}
    \end{array}
    \right.
    $$
    再根据$\theta_1+\theta_2+...\theta_k = 1$得：
    $$
    \theta_1+\theta_2+...\theta_k = 1\ \ \ \ => \ \ \ \ \left\{
    \begin{array}\\
    \lambda = -(m_1+m_2+m_3+...+m_k) = -N\\
    \theta_i = \frac{m_i}{N}
    \end{array}
    \right.
    $$
    也即：
    $$
    \hat\theta_i = \frac{m_i}{N}
    $$

  * 贝叶斯估计

    其假定$\theta_i$并非定值，受到一定限制，例如大致服从某一分布。

    对于上述例子，假设$\theta$ 服从某一先验分布，如$P(\theta) = \eta \theta_1^{\alpha_1} \theta_2^{\alpha_2} ...\theta_k^{\alpha_k}$

    则根据贝叶斯估计：
    $$
    p(\theta|Y') = \frac{p(\theta) p(Y'|\theta)}{p(Y)} \\
    \underset{\theta}{max}\ p(\theta|Y') => \underset{\theta}{max}\ p(\theta)p(Y'|\theta)\\
    其中：p(\theta) = \eta \theta_1^{\alpha}\theta_2^{\alpha}...\theta_k^{\alpha} \\
    p(Y'|\theta)也即使前面极大似然估计中l(\theta)：\\
    p(Y'|\theta) = \theta_4 \theta_1 \theta_1 .....\theta_k=\theta_1^{m_1} \theta_2^{m_2} \theta_3^{m_3}...\theta_k^{m_k}\\
    $$
    最优化问题：
    $$
    \underset{\theta}{max}\{ p(\theta)p(Y'|\theta)\}=>\underset{\theta}{max}\{\eta \theta_1^{\alpha}\theta_2^{\alpha}...\theta_k^{\alpha}\theta_1^{m_1}\theta_2^{m_2}...\theta_k^{m_k}\}=>\underset{\theta}{max}\{\theta_1^{\alpha+m_1}\theta_2^{\alpha+m_2}...\theta_k^{\alpha+m_k}\}
    $$
    该最优化为题的解，根据极大似然估计的解决步骤，可以得出：
    $$
    \hat \theta_i = \frac{m_i+\alpha}{N+k\alpha}
    $$

* 极大似然估计与贝叶斯估计的比较

  * 当样本中测试数量足够大时，即 N 足够大时，可以根据式子看出，极大似然估计与贝叶斯估计结果等价。
  * 当N很小的时候，极有可能样本中某种Y取值情况数为零，即$m_i=0$，因此会出现$\theta_i=0$，导致$l(\theta)=0$。此时最优化问题求解就会有问题。对比与极大似然估计，就防止了这种问题的发生。特别的，当$\alpha=1$时，为拉普拉斯平滑；当$\alpha=0$时，就为极大似然估计。


# 2. 具体实例与代码实现



