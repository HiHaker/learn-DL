import pickle
import torch.nn as nn
import matplotlib.pyplot as plt


def D_loss(real_op, real_lable, fake_op, fake_lable):
    return nn.BCELoss()(real_op, real_lable) + nn.BCELoss()(fake_op, fake_lable)


def G_loss(fake_op, real_lable):
    return nn.BCELoss()(fake_op, real_lable)


def show_results(imgs, col, row, data_path, save_path, desc):
    fig = plt.figure()

    # 展示图像
    imgs = imgs.cpu().detach().numpy()

    scale = 2

    figsize = (col * scale, row * scale)
    _, axes = plt.subplots(row, col, figsize=figsize)
    for i in range(row):
        for j in range(col):
            axes[i][j].imshow(imgs[i * col + j], cmap='gray')
    # 保存图像
    plt.savefig(save_path + desc + 'result.png')
    plt.show()

    # 绘制loss曲线
    with open(data_path+'lossD.txt', 'rb') as f:
        lossD = pickle.load(f)
    with open(data_path+'lossG.txt', 'rb') as f:
        lossG = pickle.load(f)
    epoch = [i for i in range(len(lossD))]
    plt.plot(epoch, lossD, color='red', label='lossD')
    plt.plot(epoch, lossG, color='yellow', label='lossG')
    # 保存图像
    plt.savefig(save_path + desc + 'loss.png')
    plt.show()
