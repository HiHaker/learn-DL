### chapter2 Linear Algebra

### 2.1 标量 向量 矩阵 张量

#### Scalars

**A scalar is a single number.** We usually give scalars lower-case variable names. For example, Let $n \in R$ be the number of the unit.

#### Vectors

**A vector is an array of numbers.** Typically we give vectors lower case names written in bold typeface.
$$
\bf x = {
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n \\
\end{bmatrix}
}
$$


#### Matrices

**A matrix is a 2-D array of numbers.** We usually give matrices upper-case variable names with bold typeface.
$$
\bf A = {
\begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22} \\
\end{bmatrix}
}
$$

#### Tensors

**In some cases we will need an array with more than two axes.** In the general case, an array of numbers arranged on a regular grid with a variable number of axes is known as a tensor. 

#### Transpose

 The transpose of a matrix is the mirror image of the matrix across a diagonal line.
$$
(A^{T})_{i,j} = A_{j,i}
$$

- **Vectors can be thought of as matrices that contain only one column.** 

- **A scalar can be thought of as a matrix with only a single entry**

#### Matrices addition

$$
\bf C = A + B （C_{i,j} = A_{i,j}+B_{i,j}, \ \ \ must \ have \ the \ same \ shape）
$$

$\bf D = $ $a \cdot \bf B$ $+c$ where $D_{i,j} = a \cdot B_{i,j} + c$, notes：$\bf B,D$ are matrices，a,c is  scalar.

#### broadcasting

$\bf C = A+b$ where $C_{i,j} = A_{i,j} + b_{j}$

### 2.2 矩阵和向量相乘

#### Matrix multiplication

$$
\bf C=AB \\
\rm C_{i,j} = \sum_{k}A_{i,k}B_{k,j}
$$

矩阵A的每一行乘以矩阵B的每一列。

##### properties

- $\bf A(B+C)=AB+AC$
- $\bf A(BC) = (AB)C$
- $\bf AB \neq BA$, not commutative!

### 2.3 单位矩阵和逆矩阵

#### Identity Matrix

**An identity matrix is a matrix that does not change any vector when we multiply that vector by that matrix.**

For example, $\bf I_{3}$, $\forall \bf x \rm \in R^{n}, I_{n}x=x$
$$
{
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
}
$$

#### Inverse Matrices

$\bf A^{-1}A=I_{n}$

now we can solve linear equation by following steps:
$$
Ax=b \\
A^{-1}Ax=A^{-1}b \\
I_{n}x = A^{-1}b \\
x=A^{-1}b
$$

### 2.4 线性相关(Linear dependence)和向量张成的空间(Span)

如果逆矩阵$A^{-1}$存在，说明线性方程组$\bf Ax=b$存在唯一解。

Linear Combination（线性组合）：$\sum_{i}c_{i}\bf v^i$

一组向量张成的空间（Span）是它们线性组合后的点组成的空间。

判断$\bf Ax=b$是否有解就是判断$\bf b$在不在线性变换$A$张成的的列空间（Column space）上。

假设要使方程组对于任意的$\bf b \in R^{m}$有解，那我们需要确保矩阵$\bf A$的列数$\ge m$，否则，会存在$\bf b$使得方程组无解（考虑假设$m=3$，说明$\bf b$可以在三维空间中任意取值， 而当$\bf A$列空间维度为2时，仅仅是一个平面，无法取到整个三维空间的值）。

为什么需要列数$\ge$呢？是因为$\bf A$的某些列可能是冗余的，对张成的列空间的维度没有贡献，比如说可能列数为4，但张成的空间的维度只有3。

#### 线性相关（Linear dependence）和线性无关

一组向量中至少有一个是多余的，没有对张成的空间有任何贡献，救说这组向量线性相关，否则称为线性无关（Linearly independent）。

要保证矩阵可逆，即方程组有唯一解，就要确保列数$\le m$，（列数在线性方程组里表示的意义是未知数的个数，$m$表示方程组的个数）否则会有无数多个解。

综上所述，列数$n=m$，也就是说$\bf A$是一个**方阵（square）**，所有列向量线性无关，线性相关的方阵被称为**奇异的（Singular）**。

如果$\bf A$不是方阵或它是奇异的，它也可能有解，但不能使用逆矩阵求解。

### 2.5 范数（Norms）

在机器学习中，我们经常使用称为**范数（norm）**的函数衡量向量的大小，$L^{p}$范数定义如下：
$$
\left\|x\right\|_p=(\sum_{i}|x_{i}|^{p})^{\frac{1}{p}} \dots, \ p \in R, \ p\ge1
$$
范数将向量映射为一个非负的函数值，向量$\bf x$的范数衡量原点到$\bf x$的距离。

当$p=2$，$L^2$范数被称为欧几里得范数，它在机器学习中十分频繁出现，被表示为$\|x\|$。

当机器学习问题中零和非零元素之间的差异非常重要时，通常会使用$L^1$范数，
$$
\|x\|_1=\sum_i|x_i|
$$
最大范数：$\|x\|_{\infty}=max_{i}|x_i|$

Frobenius范数：衡量矩阵的大小
$$
\|A\|_F=\sqrt{\sum_{i,j}A_{i,j}^{2}}
$$
两个向量的点积：
$$
x^Ty=\|x\|_2\|y\|_2cos\theta
$$

### 2.6 特殊类型的矩阵和向量

**对角矩阵**（Diagonal）：

A matrix D is diagonal if and only if $D_{i,j}=0$ for all $i \neq j$。

单位矩阵就是对角矩阵，用$diag(v)$表示一个对角方阵的对角元素由向量$\bf v$中的元素给定，计算$diag(v)x=v \odot x$。

对对角方阵的逆矩阵计算也很方便，对角方阵的逆矩阵存在，当且仅当对角元素都是非零值，$diag(v)^{-1} = diag([\frac{1}{v_1},\dots,\frac{1}{v_n}]^T)$。

**对称（symmetric）矩阵**是转置后和自己相等的矩阵：$A=A^T$

**单位向量**：$\|x\|_2=1$。

**正交**（orthonormal）：$x^Ty=0$。

**标准正交**（orthonormal）：向量不仅互相正交，且范数都为1。

**正交矩阵**：行向量列向量分别标准正交
$$
A^TA=AA^T=I,A^{-1}=A^T
$$

### 2.7 特征分解

**特征分解（eigendecomposition）**是使用最广的矩阵分解之一，即我们将矩阵分
解成一组特征向量和特征值。

方阵$A$的特征向量（eigenvector）是指与$A$相乘后相当于对该向量进行缩放
的非零向量$v$：
$$
Av=\lambda v
$$
假设矩阵$A$有$n$个线性无关的特征向量$\{v^{(1)},v^{(2)}\dots v^{(n)}\}$，对应着特征值$
\{\lambda_1,\lambda_2 \dots \lambda_n\}$。我们将特征向量连接成一个矩阵，使得每一列是一个特征向量：类似地，我们也可以将特征值连接成一个向量：$V=[v^{(1)},v^{(2)} \dots v^{(n)}]$
因此$A$的特征分解（eigendecomposition）可以记作：
$$
A=Vdiag(\lambda)V^{-1}
$$
所有特征值都是正数的矩阵被称为**正定（positive definite）**；所有特征值都是非负数的矩阵被称为**半正定（positive semidefinite）**。同样地，所有特征值都是负数的矩阵被称为**负定（negative definite）**；

### 2.8 奇异值分解

$$
A=UDV^T
$$

假设$A$是一个$m*n$的矩阵，那么$U$是一个$m*m$的矩阵，$D$是一个$m*n$的矩阵，$V$是一个$n*n$矩阵。
这些矩阵中的每一个经定义后都拥有特殊的结构。矩阵$U$和$V$都定义为正交矩阵，而矩阵$D$定义为对角矩阵。注意，矩阵$D$不一定是方阵。
对角矩阵$D$对角线上的元素被称为矩阵$A$的奇异值（singular value）。矩阵
$U$的列向量被称为左奇异向量（left singular vector），矩阵$V$的列向量被称右奇异向量（right singular vector）。
事实上，我们可以用与$A$相关的特征分解去解释$A$的奇异值分解。$A$的左奇
异向量（left singular vector）是$AA^T$的特征向量。$A$的右奇异向量（right singular vector）是$A^TA$的特征向量。$A$的非零奇异值是$A^TA$特征值的平方根，同时也是$AA^T$特征值的平方根。
SVD最有用的一个性质可能是拓展矩阵求逆到非方矩阵上。我们将在下一节中
探讨。

### 2.9 Moore-Penrose伪逆

### 2.10 迹运算

迹运算返回的是矩阵对角元素的和：
$$
\bf Tr \rm (A)=\sum_{i}A_{i,j}
$$

### 2.11 行列式

### chapter3 Probability and InformationTheory(概率和信息论)

#### 3.1 为什么使用概率？

概率论是对不确定性进行度量的一种工具，有两个流派，频率流派和贝叶斯流派。

#### 3.2 随机变量

随机变量是一个函数，将随机事件的结果映射为函数值，分为离散型随机变量和连续型随机变量。

#### 3.3 概率分布

概率分布（probability distribution）用来描述随机变量或一簇随机变量在每一个可能取到的状态的可能性大小。我们描述概率分布的方式取决于随机变量是离散的还是连续的。

离散型随机变量对应概率质量函数（PMF），连续型随机变量对应概率密度函数（PDF）。用函数$u(x;a,b)$来表示在实数区间上的均匀分布，$;$符号表示以什么为参数，这里，$x$作为自变量，$a,b$作为函数的参数，$u(x;a,b)=\frac{1}{b-a}$，用$x \sim U(a,b)$表示$x$在区间$[a,b]$上均匀分布。

#### 3.4 边缘概率

$$
P(X=x)=\sum_yP(X=x,Y=y)
$$

加和规则（sum rule）
$$
f_X(x)=\int_{-\infty}^{\infty}f(x,y)dy
$$

#### 3.5 条件概率

在很多情况下，我们感兴趣的是某个事件，在给定其他事件发生时出现的概率。这种概率叫做条件概率。我们将给定$X=x \ Y=y$发生的条件概率记为
$P(Y=y|X=x)$。这个条件概率可以通过下面的公式计算：
$$
P(Y=y|X=x)=\frac{P(Y=y, X=x)}{P(X=x)}
$$

#### 3.6 条件概率的链式法则

乘法定理：
$$
P(a,b,c)=P(a|b,c)P(b,c) \\
P(b,c)=P(b|c)P(c) \\
thus \ \ \  P(a,b,c)=P(a|b,c)P(b|c)P(c)
$$

#### 3.7 独立性和条件独立性

两个随机变量$x$和$y$，如果它们的概率分布可以表示成两个因子的乘积形式，并且一个因子只包含$x$另一个因子只包含$y$，我们就称这两个随机变量是相互独立的。
$$
p(X=x,\ \ Y=y)=p(X=x)p(Y=y)
$$
条件独立：
$$
p(X=x, Y=y|Z=z)=p(X=x|Z=z)p(Y=y|Z=z)
$$

#### 3.8 期望、方差和协方差

##### 期望

定义 设离散型随机变量$X$的分布律为$P\{X=x_k\}=p_k,k=1,2 \dots$，若级数$\sum_{k=1}^{\infty}x_kp_k$绝对收敛，则称级数$\sum_{k=1}^{\infty}x_kp_k$的和为随机变量$X$的**数学期望**，记为$E(X)$。

对于连续型随机变量$X$，设其概率密度为$f(x)$，若积分$\int_{-\infty}^{\infty}xf(x)\mathrm dx$绝对收敛，则称积分$\int_{-\infty}^{\infty}xf(x)\mathrm dx$的值为随机变量$X$的**数学期望**，记为$E(X)$。数学期望简称期望，又称为均值。

**定理**

设$Y$是随机变量$X$的函数：$Y=g(X)$（$g$是连续函数）

- 如果$X$是离散型随机变量，它的分布律为$P\{X=x_k\}=p_k,k=1,2 \dots$，若$\sum_{k=1}^{\infty}g(x_k)p_k$绝对收敛，则有

$$
E(Y)=E[g(X)]=\sum_{k=1}^{\infty}g(x_k)p_k
$$

- 如果$X$是连续型随机变量，它的概率密度为$f(x)$，若$\int_{-\infty}^{\infty}g(x)f(x)\mathrm dx$绝对收敛，则有
  $$
  E(Y)=E[g(X)]=\int_{-\infty}^{\infty}g(x)f(x)\mathrm dx
  $$

##### 方差

定义 设$X$是一个随机变量，若$E\{[X-E(X)]^2\}$存在，则称$E\{[X-E(X)]^2\}$为$X$的方差，记为$D(X)$或$Var(X)$，即：
$$
D(X)=Var(X)=E\{[X-E(X)]^2\}
$$
在应用上还引入量$\sqrt {D(X)}$，记为$\sigma(X)$，称为标准差或均方差。

由定义知，方差实际上就是随机变量$X$的函数$g(X)=(X-E(X))^2$的数学期望，

于是对于**离散型随机变量**，有
$$
D(X)=\sum_{k=1}^{\infty}[x_k-E(X)]^2p_k
$$
其中$P\{X=x_k\}=p_k,k=1,2 \dots$是$X$的分布律。

对于**连续型随机变量**，有：
$$
D(X)=\int_{-\infty}^{\infty}[x-E(X)]^2f(x)\mathrm dx
$$
其中$f(x)$是$X$的概率密度。

随机变量$X$的方差可按下列公式计算。
$$
D(X)=E(X^2)-E(X)^2
$$
**方差的几个重要性质**

- 设$C$是常数，则$D(C)=0$。

- 设$X$是随机变量，$C$是常数，则有
  $$
  D(CX)=C^2D(X), D(X+C)=D(X)
  $$

- 设$X,Y$是两个随机变量，则有
  $$
  D(X+Y)=D(X)+D(Y)+2E\{(X-E(X))(Y-E(Y))\}
  $$
  特别，若$X,Y$相互独立，则有$D(X+Y)=D(X)+D(Y)$

##### 协方差及相关系数

定义 量$E\{[X-E(X)][Y-E(Y)]\}$称为随机变量与的**协方差**，记为$Cov(X,Y)$，即
$$
Cov(X,Y)=E\{[X-E(X)][Y-E(Y)]\}
$$
而
$$
p_{XY}=\frac{Cov(X,Y)}{\sqrt{D(X)}\sqrt{D(Y)}}
$$
称为随机变量$X$与$Y$的相关系数。

##### 矩、协方差矩阵

定义 设$X$和$Y$是随机变量，若
$$
E(X^k), k=1,2\dots
$$
存在，称它为$X$的$k$阶原点矩，简称$k$**阶矩**。

若
$$
E\{[X-E(X)]^k\},k=2,3\dots
$$
存在，称它为$X$的$k$**阶中心矩**。

若
$$
E(X^kY^l),k,l=1,2\dots
$$
存在，称它为$X$和$Y$的$k+l$**阶混合矩**。

若
$$
E\{[X-E(X)]^k[Y-E(Y)]^l\}, k,l=1,2\dots
$$
存在，称它为$X$和$Y$的$k+l$**阶混合中心矩**。



设$n$维随机变量$(X_1,X_2,\dots ,X_n)$的二阶混合中心矩
$$
c_{ij}=Cov(X_i,X_j)=E\{[X_i-E(X_i)][X_j-E(X_j)]\},i,j=1,2,\dots ,n
$$
都存在，则称矩阵
$$
C={
\begin{bmatrix}

c_{11} & c_{12} & \dots & c_{1n} \\
c_{21} & c_{22} & \dots & c_{2n} \\
\dots & \dots & \dots & \dots \\
c_{n1} & c_{n2} & \dots & c_{nn} \\
\end{bmatrix}
}
$$
为$n$维随机变量$(X_1,X_2,\dots ,X_n)$的**协方差矩阵**。

#### 3.9 常用概率分布

二项分布、正态分布（高斯分布）
