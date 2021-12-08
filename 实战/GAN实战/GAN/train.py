import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from model import Generator, Discriminator

# 批大小
batch_size = 32
# 线程数
num_workers = 0

# ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，其将每一个数值归一化到[0,1]
transform = transforms.ToTensor()

# root参数是数据存放路径，train参数是是否是训练集，transform参数是数据的转换方式
train_data = datasets.MNIST(root='../../data/MNIST', train=True,
                            download=False, transform=transform)
test_data = datasets.MNIST(root='../../data/MNIST', train=False,
                           download=False, transform=transform)
# 创建dataloader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

# 加载模型
G = Generator()
D = Discriminator()
# 指定loss函数
criterion = nn.BCELoss()
# 学习率
learning_rate = 1e-5
# 优化器
optimizerG = optim.SGD(G.parameters(), lr=learning_rate)
optimizerD = optim.SGD(D.parameters(), lr=learning_rate)

# 训练轮次
epoch = 20
for i in range(epoch):
    # 训练
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
        noise = torch.normal(0, 1, (batch_size, 10))

        # 图片打平
        images = torch.flatten(images, start_dim=1)
        fake_img = G(noise)
        # Generator 产生的数据的标签为0
        fake_label = torch.zeros([batch_size])
        imgs_label = torch.ones([batch_size])
        total_img = torch.cat((fake_img, images), dim=0)
        total_label = torch.cat((fake_label, imgs_label), dim=0)
        outputs = D(total_img)
        outputs = torch.flatten(outputs)
        lossD = criterion(outputs, total_label)
        lossD.backward()
        optimizerD.step()

        # 固定住D，训练G
        D.eval()
        G.train()
        # 产生高斯噪声数据
        noise = torch.normal(0, 1, (batch_size, 10))
        fake_img = G(noise)
        outputs = D(fake_img)
        outputs = torch.flatten(outputs, start_dim=1)
        lossG = 1-torch.log(outputs).sum()
        lossG.backward()
        optimizerG.step()

    print('epoch: %d--lossD: %.3f, lossG: %.3f.' % (i, lossD, lossG))
