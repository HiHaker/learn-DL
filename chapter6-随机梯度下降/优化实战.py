import torch
import numpy as np
import matplotlib.pyplot as plt


# 定义待优化的函数
def himmelblau(data):
    return (data[0] ** 2 + data[1] - 11) ** 2 + (data[0] + data[1] ** 2 - 7) ** 2


# 数据准备
def prepareData():
    # -6~6，步长为0.1
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau([X, Y])
    return X, Y, Z


def plotImg():
    fig = plt.figure("himmelblau")
    ax = fig.gca(projection="3d")
    x, y, z = prepareData()
    ax.plot_surface(x, y, z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def optimFunc():
    x = torch.tensor([0, 0.], requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=1e-3)

    for step in range(20000):

        pred = himmelblau(x)

        optimizer.zero_grad()
        pred.backward()
        optimizer.step()

        if step % 2000 == 0:
            print("step {}: x={}, f(x)={}".format(step, x.tolist(), pred.item))


# plotImg()
optimFunc()