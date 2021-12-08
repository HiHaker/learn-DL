### 回顾

上次课中讲解了通过一个线性分类器，我们得到了一张图像对应的每个类别的得分：

![](.\img\l2_1.png)

接下来要做的事就是：

- 定义一个loss函数（也称为代价函数或目标函数）量化计算出的得分和真实数据标签的差异。

- 想办法高效的找到使损失函数最小的参数的方法。

### Loss Function

下图是一个简单的例子，只包含3个类别的图像。

![](.\img\l2_2.png)

损失函数定义如下：整个训练集的损失是对训练数据的所有样本的损失的平均。
$$
L = \frac{1}{N}\sum_iL_i(f(x_i,W),y_i)
$$

#### Multiclass Support Vector Machine loss（SVM loss）

$$
L_i = \sum_{j \ne y_i}
\begin{cases}
0 \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ if \ s_{y_{i}} \ge s_{j}+\Delta \\ 
s_{j} - s_{y_i} + \Delta \ \ \ \ \ otherwise
\end{cases}
$$

或者表示为：
$$
L_i = \sum_{j \ne y_i}max(0, s_j-s_{y_i}+\Delta)
$$
这个损失函数也被称为合页损失（hinge loss），因为其函数图像很像书本的合页，这是因为随着$s_{y_i}-s_j$的值的减小，这个loss是逐渐降低的，一旦$s_{y_i}-s_j$的值低于了某一个阈值（$\Delta$），函数的值就变成0了。SVM loss期望的是分类器得分函数对图像所属正确类别的得分至少要大于$\Delta$，如果不是的话，loss就不会是0，就会累加。

根据之前介绍的线性分类器，可以把loss写成下面的形式：$w_j$是$W$的第$j$行。
$$
L_i = \sum_{j \ne y_i}max(0, w_jx_i-w_{y_i}x_i+\Delta)
$$
![](.\img\l2_3.png)

```python
def loss_func(scores, y):
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i
```

对于整个数据集的样本：
$$
L = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \ne y_i}max(0, w_jx_i-w_{y_i}x_i+\Delta)
$$

#### 正则化（Regularization）

假定我们找到了一个权重参数矩阵$W$使得$L=0$，那么$W$是唯一的吗？

答案是不唯一，$2W$也可以使得$L=0$。证明如下：
$$
s_j-s_{y_i} + \Delta \lt 0 \ \ (1) \\ 
s_j + \Delta \lt s_{y_i} \ \ (2) \\
2(s_j + \Delta) \lt 2s_{y_i} \ \ (3) \\
2s_j + \Delta \lt 2s_{y_i} \ \ (4)
$$
$L=0$说明(1)式成立，进一步推得(2)、(3)式成立，当权重参数变为原来的2倍后，可知$s_j \ s_{y_i}$也变为原来的两倍，所以有(4)式成立，而(3)式可以推得(4)式，即可证明。

既然$W$不唯一，那么分类器要如何在这些$W$值中做出选择呢？

事实上，目前的loss函数仅仅告诉了分类器该怎么样去拟合训练数据，而实际上，我们不关心这个分类器对训练数据的拟合，我们更关心分类器在测试数据上的表现。

为了不让模型过度的拟合，我们会在loss函数中加上一个正则项

![](.\img\l2_4.png)

比较常用的是平方L2范数（通过对所有参数的二次惩罚来阻止一个较大的权重）：
$$
R(W)=\sum_k \sum_l W_{k,l}^{2}
$$
现在，损失函数的完全形式为：
$$
L = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \ne y_i}[max(0, f(x_i;W)_j-f(x_i;W)_{y_i}+\Delta)] + \lambda \sum_k \sum_l W_{k,l}^2
$$
其他的一些正则项：

![](.\img\l2_5.png)

因为偏置$b$并不控制输入维度的影响力，不像权重$W$，所以在实际应用中，一般仅仅正则化权重$W$。

#### Softmax Classifier（Multinomial Logistic Regression）

在之前介绍的multiclassSVM中，对scores没有做过多的解释说明，只是期望其正确类别的得分大于其错误类别的得分，而在这里介绍的Softmax分类器中，它给出了一个更为直观的输出（归一化的类别概率），其中，函数映射$f(x_i;W)=Wx_i$不变，但将分数解释为了每一个类别的**非归一化对数概率**，并且以**交叉熵损失（cross-entropy loss）**代替了原来的**合页损失（hinge loss）**。

前面提到我们将分数解释为**非归一化对数概率**，所以现在我们通过Softmax函数来对分数进行归一化，首先进行**exp**操作（指数化），让他们都为正数，接着进行归一化，让他们相加等于1，当分数经过Softmax函数之后，就得到了概率分布。
$$
P(Y = k|X=x_i)=\frac{e^{s_k}}{\sum_j e^{s_j}}, \ \ \ where \ \ s=f(x_i;W)
$$
损失函数为：
$$
L_i=-logP(Y=y_i|X=x_i) \\
L_i=-log(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}})
$$
最大化对数概率，或者最小化负对数概率。

```python
def Softmax_loss_func(scores, y):
    return -np.log(np.exp(scores[y])/np.sum(np.exp(scores)))
```

![](.\img\l2_6.png)

------

### Optimization

在之前我们介绍了**得分函数（score function）**、**损失函数（loss function）**，接下来将介绍第三个也是最后一个关键的部分：**最优化**。最优化是去找到使损失函数最小的权重参数$W$的过程。

一旦我们理解了这三个部分是怎么互相作用之后，我们就要回到第一个部分（参数化函数映射）：首先介绍神经网络，然后是卷积神经网络，损失函数和最优化过程相对来说不变。

#### 策略1 随机搜索

```python
import numpy as np

# Softmax loss
def Softmax_loss_func(scores, y):
    return -np.log(np.exp(scores[y])/np.sum(np.exp(scores)))

# 100张训练数据 simple case
N = 100
# 5个类别
class_num = 5
img_size = 32
# 训练数据
img = np.random.random(size=[N, img_size*img_size])
# 标签
label = np.random.randint(0,5,size=N)
best_loss = float('inf')
# 随机搜索1000次
for j in range(1000):
    total_loss = 0
    # 记录预测正确的次数    
    count = 0
    # 随机搜索    
    W = np.random.random(size=[class_num, img_size**2])
    for i in range(N):
        x_i = img[i]
        scores = W.dot(x_i)
        predict_label = np.argmax(scores)
        if predict_label == label[i]:
            count += 1
        loss = Softmax_loss_func(scores, label[i])
        total_loss += loss
    # 准确率    
    accuracy = count/N
    # 整个数据集的损失    
    loss = total_loss/N
    if loss < best_loss:
        best_loss = loss
        print('Best loss updated in epoch{}: {}'.format(j, best_loss))
        print('accuracy: {}'.format(accuracy))
```

![](.\img\l2_7.png)

可以看出来，效果很不好。

##### 随机局部搜索

```python
# 初始W值
W = np.random.random(size=[class_num, img_size**2])
# print(W)
# 随机搜索1000次
for j in range(1000):
    # 随机局部搜索    
    W = W + np.random.randn(class_num, img_size**2)*0.1
```

效果更差，取决于设定的步长，太大了很容易就收敛不了

#### 策略2 朝着下坡的方向前进

在一维空间中，导数的定义如下：
$$
\frac{df(x)}{dx}=\lim_{h \to 0}\frac{f(x+h)-f(x)}{h}
$$
在多维空间中，梯度是沿着每个维度方向的偏导数组成的向量。

梯度的方向是该点函数值增长最快的方向，所以负梯度的方向就是函数值下降最快的方向。

#### 计算梯度

##### 数值近似 - 简单，计算出的是近似值，计算慢

![](.\img\l2_8.png)

```python
import numpy as np

# Softmax loss
def Softmax_loss_func(scores, y):
    return -np.log(np.exp(scores[y])/np.sum(np.exp(scores)))

# 10张训练数据 simple case
N = 10
# 5个类别
class_num = 5
img_size = 4
# 训练数据
img = np.random.random(size=[N, img_size*img_size])
# 标签
label = np.random.randint(0,5,size=N)
best_loss = float('inf')
# 初始W值
W = np.random.random(size=[class_num, img_size**2])
# 步长
step_size = 0.005
# 梯度下降100次
for epoch in range(100):
    total_loss = 0
    # 记录预测正确的次数    
    count = 0
    # 梯度下降
    for i in range(N):
        x_i = img[i]
        scores = W.dot(x_i)
        predict_label = np.argmax(scores)
        if predict_label == label[i]:
            count += 1
        loss = Softmax_loss_func(scores, label[i])
        total_loss += loss
    
    grad = np.zeros(shape=[class_num, img_size**2])
    for j in range(class_num):
        for k in range(img_size**2):
            W_new = W
            total_loss_new = 0
            # 改变其中一个参数的值             
            W_new[j][k] += 0.01
            # 重新计算loss            
            for n in range(N):
                x_n = img[n]
                scores = W_new.dot(x_n)
                predict_label = np.argmax(scores)
                loss = Softmax_loss_func(scores, label[n])
                total_loss_new += loss
            # 计算梯度
            grad[j][k] = (total_loss_new - total_loss)/0.01
    # 梯度下降
    W -= step_size*grad
    
    # 准确率    
    accuracy = count/N
    # 整个数据集的损失    
    loss = total_loss/N
    print('loss: {}, accuracy: {} of epoch{}.'.format(loss,accuracy,epoch))
```

![](.\img\l2_9.png)

我们从图中可以看到loss先有了一点下降，随后就一直在增大，这和我们设置的步长有关。梯度只是告诉了我们哪个方向函数值增加最快，但没有告诉我们要走多远。

我们从程序中可以看到，使用这种数值差分近似的方法进行计算非常耗费时间。它的优点是简单，但是缺点是我们得到的值只是近似值，不是真实的梯度值，而且计算非常耗费时间。

##### 利用微积分计算梯度（链式法则） - 精确，容易出错

第二种方法是微积分，它能够让我们直接推导出一个梯度的公式。

![](.\img\l2_10.png)

在实际操作时，一般不去用整个训练集计算梯度（数据集太大了），而是用minibatch的一块数据进行计算，可以近似我们的梯度，称为**随机梯度下降**（原来的意思单指代仅使用一个数据进行计算的情况，现在一般把minibatch的也称为SGD）。

相对于将图像的原始像素直接送入到线性分类器计算一个得分，我们更可能会首先计算图像的特征，再将图像的特征送入这个线性分类器，这样效果会更好。

下面的例子展示了通过将点的坐标转换为极坐标表示就可以很好的区分这两类点。

![](.\img\l2_11.png)

