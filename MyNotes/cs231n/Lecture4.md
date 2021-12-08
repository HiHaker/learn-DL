### 反向传播

结合3blue1brown的视频复习，手推反向传播。

#### 问题定义

给定一个函数$f(x)$，$x$作为向量输入，我们想要计算$f(x)$在$x$处的梯度$\nabla f(x)$。

#### 回顾

回顾前面的内容，我们定义了一个**线性分类器**，对应一个得分函数$f(x_i,W)$，为了量化这个线性分类器分类的能力有多糟糕，我们定义了**损失函数**，介绍了两种常用的损失函数：**multiclassSVM Loss**以及**Softmax Loss**。为了提升分类器能力，我们介绍了**最优化**的方法，期望找到使得损失函数值最小的参数。接下来，我们介绍了**梯度下降**方法，以最快地速度减少损失函数的值，而计算梯度，可以使用差分近似方法，优点是计算简单，缺点是值是近似的，计算过程需要耗费大量的时间

在机器学习中，我们认为输入的训练数据$x_i$是给定且固定的，权重参数是我们控制的变量，所以一般我们不去求$x_i$的梯度（除了在特殊情况下：可视化以及解释神经网络可能是怎么做的）。

#### 一些简单的例子

- $f(x,y)=xy, \ \ \ \to \ \ \ \frac{\partial f}{\partial x}=y, \ \frac{\partial f}{\partial y}=x, \ \ \nabla f=[\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}] $

- $f(x,y)=x+y, \ \ \ \to \ \ \ \frac{\partial f}{\partial x}=1, \ \frac{\partial f}{\partial y}=1$

- $f(x,y)=max(x,y), \ \ \ \to \ \ \ \frac{\partial f}{\partial x}=1(x \ge y), \ \frac{\partial f}{\partial y}=1(y \ge x)$

下面来看更复杂一些的，复合函数：
$$
f(x,y,z)=(x+y)z
$$
令$q=(x+y)$，我们就得到$f=qz$，而$\frac{\partial f}{\partial q}=z$，$\frac{\partial f}{\partial z}=q$，又因为$\frac{\partial q}{\partial x}=1$，$\frac{\partial q}{\partial y}=1$，我们并不关心设的中间值$q$，我们关心$\frac{\partial f}{\partial x}$，$\frac{\partial f}{\partial y}$，根据链式法则：$\frac{\partial f}{\partial x}=\frac{\partial f}{\partial q}\frac{\partial q}{\partial x}$，$\frac{\partial f}{\partial y}=\frac{\partial f}{\partial q}\frac{\partial q}{\partial y}$，在实践中，这只是两个数的乘法。

#### 计算图（Computational graph）

是一个用于计算任意复杂函数的解析梯度的框架：

![](.\img\l3_1.png)

从一个简单的例子说起：

![](.\img\l3_2.png)

绿色部分的字代表变量的值，红色部分的字代表$f$关于变量的偏导数（为了简化一般也称梯度）。

**前向传播（计算函数值）：**根据变量的值从左往右进行计算，$x=-2, \ y=5, \ z=-4$，$x+y=3$，$q*z=-12$.

**反向传播（计算梯度）：**从右往左，从最前一个结点开始往后计算梯度。

- 因为$\frac{\partial f}{\partial f}=1$，$*$这个结点输出值梯度为1

- $+$结点，令其输出值为$q$，因为$f=qz$，局部梯度为$\frac{\partial f}{\partial q}=z$，从前面传过来的梯度为1，所以结点的输出值梯度为$z*1=z=-4$
- $z$结点输出值的局部梯度为$\frac{\partial f}{\partial z}=q$，从前面传过来的梯度为1，所以结点梯度为$q*1=q=3$
- $x$结点输出值的局部梯度为$\frac{\partial q}{\partial x}=1$，从前面传过来的梯度为-4，所以结点梯度为$1*-4=-4$
- $y$结点输出值的局部梯度为$\frac{\partial q}{\partial y}=1$，从前面传过来的梯度为-4，所以结点梯度为$1*-4=-4$

##### 模块化，以Sigmoid函数为例

前置知识：一些简单函数的导数：
$$
f(x) = \frac{1}{x} \to \frac{df}{dx}=-\frac{1}{x^2} \\
f(x)=c+x \to \frac{df}{dx}=1 \\
f(x) = e^x \to \frac{df}{dx}=e^x \\
f(x) = ax \to \frac{df}{dx}=a 
$$
下面是一个稍微复杂的函数：
$$
f(w,x) = \frac{1}{1+e^{-(w_0x_0+w_1x_1+w_2)}}
$$
计算图如下：

![](.\img\l3_3.png)

- 因为$\frac{\partial f}{\partial f}=1$，所以$1/x$这个结点梯度为1
- $+1$结点的输出值的局部梯度为$-\frac{1}{x^2}$，从前面传过来的梯度为1，所以梯度值为$-\frac{1}{1.37^2}=-0.53$

- $exp$结点的输出值的局部梯度为$1$，从前面传过来的梯度为$-0.53$，所以梯度值为$1*-0.53=-0.53$

- $*-1$结点的输出值的局部梯度为$e^x$，从前面传过来的梯度为$-0.53$，所以梯度值为$e^{-1}*-0.53=-0.20$
- $+$结点的输出值的局部梯度为$-1$，从前面传过来的梯度为$-0.20$，所以梯度值为$-1*-0.20=0.20$

- $+$结点的输出值的局部梯度为$1$，从前面传过来的梯度为$0.20$，所以梯度值为$1*0.20=0.20$

- $w_2$结点的输出值的局部梯度为$1$，从前面传过来的梯度为$0.20$，所以梯度值为$1*0.20=0.20$

- $*$结点的输出值的局部梯度为$1$，从前面传过来的梯度为$0.20$，所以梯度值为$1*0.20=0.20$

- $*$结点的输出值的局部梯度为$1$，从前面传过来的梯度为$0.20$，所以梯度值为$1*0.20=0.20$

- $w_0$结点的输出值的局部梯度为$x_0=-1$，从前面传过来的梯度为$0.20$，所以梯度值为$-1*0.20=-0.20$

- $x_0$结点的输出值的局部梯度为$w_0=2$，从前面传过来的梯度为$0.20$，所以梯度值为$2*0.20=0.40$

- $w_1$结点的输出值的局部梯度为$x_1$，从前面传过来的梯度为$0.20$，所以梯度值为$-2*0.20=-0.40$

- $x_1$结点的输出值的局部梯度为$w_1$，从前面传过来的梯度为$0.20$，所以梯度值为$-3*0.20=-0.60$

在这个计算图中，我们注意到，划框的那几个结点组合起来就是sigmoid函数：
$$
f_{sigmoid}=\frac{1}{1+e^{-x}} \\
\frac{df}{dx}=\frac{e^{-x}}{(1+e^{-x})^2}=(\frac{1+e^{-x}-1}{1+e^{-x}})(\frac{1}{1+e^{-x}})=(1-\sigma(x))\sigma(x)
$$
这样的话，$+$结点的输出值的局部梯度为$(1-\sigma(1))\sigma(1)=0.2$，所以梯度值为$0.2*1=0.2$

所以如果某个函数的导数容易求解，我们可以随时将结点组合，组成一个函数结点进行反向传播计算。从计算图中，我们还可以看出，对于*+*结点，它将梯度值不变地反向传播；对于$*$结点，它将传过来的梯度值分别乘以对应的另一个变量后继续反向传播。

#### 向量化操作的梯度

之前介绍的部分，我们介绍的是单个变量，不过所有的概念都很直接地可以扩展到矩阵和向量操作，但我们需要关注矩阵的维度以及转置运算。

![](.\img\l3_4.png)

局部梯度变成了雅可比矩阵（Jacobian matrix），是$z$关于所有$x$的偏导数组成的矩阵，维度可以推导出来，因为$\frac{\partial L}{\partial x}$的维度和$x$的维度相同，而$\frac{\partial L}{\partial x}=\frac{\partial L}{\partial z}\frac{\partial z}{\partial x}$。

![](.\img\l3_5.png)

继续进行推导：

- 因为$\frac{\partial f}{\partial f}=1$，所以$L2$这个结点输出值梯度为1

- 首先令$q=Wx$，$*$结点的输出值的局部梯度为$\frac{\partial f}{\partial q}$，需要注意的是现在$q$是向量，$q=[q_1,q_2]^T$，所以$\frac{\partial f}{\partial q}=[\frac{\partial f}{\partial q_1},\frac{\partial f}{\partial q_2}]^T$，而$f=||q||^2=(q_1^2+q_2^2)$，所以：

$$
\frac{\partial f}{\partial q}=[\frac{\partial f}{\partial q_1},\frac{\partial f}{\partial q_2}]^T=[2q_1,2q_2]^T=2[q_1,q_2]^T=2q
$$

而从前面传过来的梯度为$1$，所以梯度值为$2q$

- $W$结点的输出值的局部梯度为$\frac{\partial q}{\partial W}$，

$$
\frac{\partial q_k}{\partial W_{i,j}}=1_{k=i}x_j \\
\begin{align}
\frac{\partial f}{\partial W_{i,j}}&=\sum_k\frac{\partial f}{\partial q_k}\frac{\partial q_k}{\partial W_{i,j}} \\
&= \sum_k(2q_k)(1_{k=i}x_j) \\
&=2q_ix_j
\end{align}
$$

- $x$结点的输出值的局部梯度为$\frac{\partial q}{\partial x}$，

$$
\frac{\partial q_k}{\partial x_i}=W_{k,i} \\
\begin{align}
\frac{\partial f}{\partial x_i}&=\sum_k\frac{\partial f}{\partial q_k}\frac{\partial q_k}{\partial x_i} \\
&= \sum_k(2q_k)W_{k,i} \\
\end{align}
$$

#### 总结

- 神经网络会非常庞大，而如果我们手写所有梯度的公式是不切实际的
- **反向传播**，沿着计算图递归地应用链式法则来计算所有的输入、参数以及中间值的梯度。
- 具体实现是一个图结构，其中的每一个结点都实现了 forward和backward这两个API。

- forward：计算运算的结果，并将任何所需的中间结果保存在内存中
- backward：运用链式法则计算loss函数对应于输入的梯度。

