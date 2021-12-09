import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from model import Generator, Discriminator

# 批大小
batch_size = 32
# 线程数
num_workers = 4
# GPU
torch.cuda.set_device(3)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，其将每一个数值归一化到[0,1]
transform = transforms.ToTensor()

# root参数是数据存放路径，train参数是是否是训练集，transform参数是数据的转换方式
train_data = datasets.MNIST(root='./dataset/MNIST', train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root='./dataset/MNIST', train=False,
                           download=True, transform=transform)
# 创建dataloader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

# 加载模型
G = Generator()
G.to(device)
D = Discriminator()
D.to(device)
# 指定loss函数
criterion = nn.BCELoss()
# 学习率
learning_rate = 1e-4
# 优化器
# optimizerG = optim.SGD(G.parameters(), lr=learning_rate)
# optimizerD = optim.SGD(D.parameters(), lr=learning_rate)
optimizerG = optim.Adam(G.parameters(), lr=learning_rate)
optimizerD = optim.Adam(D.parameters(), lr=learning_rate*0.08)

# 模型保存路径
model_savpath = './model/'
# 数据保存路径
data_savepath = './data/'

# 训练轮次
epoch = 100
total_lossD = []
total_lossG = []
for i in range(epoch):
    # 训练
    avg_lossD = 0
    avg_lossG = 0
    for step, data in enumerate(train_loader, start=1):
        # 清空梯度
        optimizerG.zero_grad()
        optimizerD.zero_grad()

        # 取出数据
        images, _ = data
        # 固定G train D
        G.eval()
        D.train()

        # 产生高斯噪声数据
        noise = torch.normal(0, 1, (batch_size, 100))

        # 图片打平
        images = torch.flatten(images, start_dim=1)
        fake_img = G(noise.to(device))

        # Generator 产生的数据的标签为0
        fake_label = torch.zeros([batch_size])
        imgs_label = torch.ones([batch_size])

        # concat
        total_img = torch.cat((fake_img, images.to(device)), dim=0)
        total_label = torch.cat((fake_label, imgs_label), dim=0)

        outputs = D(total_img.to(device))
        outputs = torch.flatten(outputs)
        lossD = criterion(outputs, total_label.to(device))
        lossD.backward()
        optimizerD.step()

        # 固定住D，训练G
        D.eval()
        G.train()

        # 产生高斯噪声数据
        noise = torch.normal(0, 1, (batch_size, 100))

        fake_img = G(noise.to(device))
        outputs = D(fake_img)
        outputs = torch.flatten(outputs, start_dim=1)
        lossG = -torch.log(outputs).sum()
        lossG.backward()
        optimizerG.step()

        avg_lossD += lossD.item()
        avg_lossG += lossG.item()

    avg_lossD /= step
    avg_lossG /= step
    # print(type(lossD.item()))
    total_lossD.append(avg_lossD)
    total_lossG.append(avg_lossG)
    print('epoch: %d--lossD: %.3f, lossG: %.3f.' % (i+1, avg_lossD, avg_lossG))

    # 每隔20轮保存一下模型
    if (i+1) % 1 == 0:
        print('save model...')
        torch.save(G.state_dict(), model_savpath+'G{}.pth'.format(i+1))
    #     # torch.save(D.state_dict(), model_savpath+'D{}.pth'.format(i+1))
with open(data_savepath+'lossD.txt', 'wb') as f:
    pickle.dump(total_lossD, f)
with open(data_savepath+'lossG.txt', 'wb') as f:
    pickle.dump(total_lossG, f)
