# 导入相关的包
import numpy as np
from math import sqrt
from collections import Counter

# KNN分类器
def KNN_classify(K, X_train, y_train, x):
    # 设置断言，k值必须在1到训练集的个数之间
    assert 1 <= K <= X_train.shape[0], "K must be valid"
    # 标签数据的数量应该等于数据的数量
    # 反斜杠为续行符
    assert X_train.shape[0] == y_train.shape[0], \
        "The size of X_train must equal to the size of y_train!"
    # 测试数据的特征数量要等于训练数据的特征数量
    assert X_train.shape[1] == x.shape[0], \
        "The feature number of x must be equal to X_train"

    # 计算出测试数据和所有训练数据之间的距离
    distances = [sqrt(np.sum((x-i)**2)) for i in X_train]
    sort_distances = np.argsort(distances)

    nearest_points = [y_train[i] for i in sort_distances[:K]]
    votes = Counter(nearest_points)

    return votes.most_common(1)[0][0]