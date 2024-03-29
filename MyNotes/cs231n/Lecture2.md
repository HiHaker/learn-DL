### 图像分类

图像在计算机内的表示是一个个数字组成的一个多维数组，对于彩色图像来说，它有3个通道（RGB），尺寸为$Width \times Height \times 3$，其中每个元素都是一个范围在$[0,255]$的数字，白色为255，黑色为0。

图像分类的任务就是给定这样一张图像，返回一个label，或者是一个label的分布概率，表明它可能是哪一个类别。

![](.\img\l1_1.png)

#### 面临的挑战

- 光照、变形、视角转移、缩放变化、遮挡、背景杂斑、同类别的变动

![](.\img\l1_2.png)

#### 数据驱动的方法

对于图像分类来说，不像对数组进行排序那样，我们可以写一个分类算法，

```python
def classify_image(image):
    # some magic code here?
    return label
```

对于这个问题来说，没有很明显的方法来进行硬编码一个算法用来识别猫。

因此，我们采用的方法是通过给计算机每个类别物体的大量的实例图像，并开发一个可以“看”这些实例并学习到关于每个类的视觉特征的学习算法，这种方法就是数据驱动的方法。

```python
def train(images, labels):
    # learning
    return model

def predict(model, test_images):
    # Use model to predict labels
    return test_labels
```



#### 图像分类的pipeline

- **输入：**输入包含了$N$张图片的集合，每张图片都被打上了属于某类的一个标签（总共有$K$类），一般把这个集合称为 **训练集** 。
- **学习：**目标是使用训练集去学习每个类别中的图像应该是什么样的，一般称这个过程为训练一个分类器或训练一个模型。
- **评估：**最后，通过给这个分类器“看”一些它从没看过的图像，比较分类器预测的标签和图像真实的标签（ground truth），评估它分类的质量。

#### 引例：最近邻分类器

在训练过程，它“记住”（存储）所有的训练数据和标签；在预测过程，它计算测试图片和训练数据的“距离”，给出和最相似的训练数据对应的标签。

##### L1距离

$$
d_1(I_1,I_2) =\sum_p |I_{1}^{p}-I_{2}^{p}|
$$

![](.\img\l1_3.png)

下面是Nearest Neighbor分类器对图像进行分类的代码：

![](.\img\l1_4.png)

可以看出，在训练过程，时间复杂度为$O(1)$，在测试过程，时间复杂度为$O(N)$，这与我们的需求相反，我们想要它在分类图片时尽可能快，训练过程多花一点时间也无所谓。

对数据间距离的测度常用的还有L2距离：
$$
d_2(I_1,I_2) =\sqrt { \sum_p (I_{1}^{p}-I_{2}^{p})^2}
$$

##### L1距离 vs L2距离：

![](.\img\l1_5.png)

我们发现L1距离取决于坐标系，如果我们改变坐标系，就会改变L1距离，但对L2距离没有影响。所以如果向量中的一些值有特殊的意义，用L1距离更合适。

#### k-nearest neighbor

![](.\img\l1_6.png)

通过选取距离最近的top-k个训练数据的类别来决定测试对象的类别，而不是只选取最近的，这就是k近邻算法，从图中我们可以看到，5-NN（k等于5）时，不同图像区域的边界更加平滑。

关于k值的选择，k值为1时，模型的复杂度达到最大；随着k值的增大，模型也趋于简单。

#### 超参数

对于k近邻算法来说，我们应该选取哪个**k值**？选取哪个**距离度量**？这些选择是**超参数**，由我们设置的，而不是学习得到的。很依赖于具体问题，必须把他们都试一遍看哪种效果最好。

关于调整超参数，需要明确的一点是我们只有在最后的最后才能使用**测试集**，测试集很珍贵，如果提前让模型“看到它”，就相当于作弊了，训练出来的模型真实表现就无从得知。

为此，我们将训练数据中的一部分划分出来，作为**验证集**，其余数据作为**训练集**。通过模型在验证集的表现来选择超参数，最后在测试集上评估我们模型的表现。

##### 交叉验证

将训练数据均匀地分为$n$等份，每次取其中一份作为验证集，其余部分作为训练集，最后将验证集的结果平均。

在实际应用中，几乎不使用交叉验证，因为计算太耗费时间了。

#### Nearest Neighbor 分类器的优缺点

优点：简单，容易实现

缺点：模型要耗费大量的存储空间，且每一次分类要耗费大量的时间，无法处理一些高维数据的分类，比如说图像这种高维数据，下面的图像和左边的原图的L2距离都是相等的。

![](.\img\l1_7.png)

------

### 线性分类 Linear Classification

搭建神经网络就像是在搭积木，把不同的组件组合，搭建出特定的神经网络。而最基本的构建块就是**线性分类器**。

#### 参数的方法（Parametric Approach）

包含两个最主要的部分：得分函数（score function）：将原始数据映射到类别得分、损失函数（loss function）：量化预测的得分与真实标签之间的差异。

#### 参数化从图像到标签分数的映射

第一个部分，就是去定义一个 score function，将一个图像的像素值，映射到对每一个类的自信分数。比如输入的是一张猫的图像，通过这个score function，得到的猫的得分为60，狗的得分为38，马的得分为79等等。

考虑一个具体的例子：

现在有一个图像的训练集$x_i \in R^D$，具有相应的标签$y_i$，$i=1,...,N, \ \ y_i \in \{1,...,K\}$

也就是说，我们有$N$个实例，每个实例的维度为$D$，每个实例都属于$K$个类别中的一个。

得分函数$f:R^D \to R^K$。

对于CIFAR-10数据集来说，$N=50,000$，$D=32 \times 32 \times3=3072$，$K=10$。

**线性分类器：**
$$
f(xi,W,b)=Wx_i+b
$$
上述等式中，假定$x_i$是打平的一个列向量，维度为$[32\times 32\times 3=3072，1]$，也就是$[D\times1]$，$W([K\times D])$和$b([K\times 1])$是函数的参数，所以有3072个数字输入函数，输出$K$个数字。$W$称为权重矩阵，$b$被称为偏置。

notes:

- $Wx_i$有效地同时评估了10个分类器，每个分类器都是$W$的一行。
- 这个方法的优点在于当训练数据被用于学习了参数$W,b$之后，我们就可以丢弃整个训练集，保留学习的参数就好了，这样一张新的测试图像可以通过这个分类器获得对应的得分。而且，我们看出，相比之前介绍的最近邻和k-最近邻算法，测试新图像的过程仅仅包括一次矩阵乘法和一次加法，而不是和所有训练数据进行比较。

![](.\img\l1_8.png)

我们可以将图像类比为高维空间中的点，$W$的每一行是一个分类器，对应图中的线，$b$，偏置让当$x_i=0$时这些线不过原点。

![](.\img\l1_9.png)

对权重矩阵$W$的另外一个解释是$W$的每一行都对应一个模板（原型），将$W$里的每个模板和图像做点积，也就是模板匹配，就得到了对应一张图像的每个类别的得分。

![](.\img\l1_10.png)

也可以换种角度思考，我们仍然在有效地做最近邻匹配，但我们不是把测试图像和所有的训练图像做匹配，只是和一张学习后的图像做匹配（它不来自训练数据，而是由训练数据学习得到的），距离测度我们使用的是点积，而不是L1或L2距离。

我们可以从上面的图像中看出，对于horse这个类的模板，似乎它有两个头，一个在左边，一个在右边，说明训练数据中存在两个方向的马；而对于car这个类的模板，它是红色的，这说明训练数据中大部分的车是红色的。当前这个线性分类器很weak，无法区分不同颜色的车，但在后面介绍的神经网络可以解决这个问题。稍微展望一下，神经网络在它的隐藏层能够检测出不同的车的类型，比如是红色，还是蓝色的，是大车，还是小车；接着，下一层的神经元将上层检测出的得分进行加权，得到一个更加准确的车辆的得分。

#### bias trick

![](.\img\l1_11.png)

可以将$b$一起放入权重矩阵$W$中，减少运算步骤。

![](.\img\l1_12.png)

经过上面的介绍，我们知道了通过一个得分函数把图像像素映射为每一个类别的得分，那么问题来了，我们要怎么判断这个分数对应的$W$是好还是坏呢？就需要定义一个损失函数。

