import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
# from vanillaGAN.model import Generator, Discriminator
from DCGAN.model import Generator, Discriminator
from utils import D_loss, G_loss

# 当前训练的模型名称
# current_model_name = './vanillaGAN/'
current_model_name = './DCGAN/'
# 批大小
batch_size = 64
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
z_dim = 100
G = Generator(z_dim)
G.to(device)
D = Discriminator()
D.to(device)
# 指定loss函数
criterion = nn.BCELoss()
# 学习率
learning_rate = 0.0002
# 优化器
# optimizerG = optim.SGD(G.parameters(), lr=learning_rate
# optimizerD = optim.SGD(D.parameters(), lr=learning_rate)
optimizerG = optim.Adam(G.parameters(), lr=learning_rate)
optimizerD = optim.Adam(D.parameters(), lr=learning_rate)
# interval = [1,10]
# scheduler = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=interval, gamma=0.1)

# 模型保存路径
model_savpath = current_model_name + 'model/'
# 数据保存路径
data_savepath = current_model_name + 'data/'

# 训练轮次
epoch = 50
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

        # 训练D
        # 产生高斯噪声数据
        noise = torch.normal(0, 1, (batch_size, z_dim))
        noise = noise.to(device)

        # 生成图像
        fake_imgs = G(noise)
        # 取出数据
        real_imgs, _ = data

        real_op = D(real_imgs.to(device))
        real_label = torch.ones_like(real_op)

        # 注意这里fake_img的detach，因为这里我们只更新D，不更新给，所以这里fake_img不参与梯度计算
        fake_op = D(fake_imgs.detach())
        fake_label = torch.zeros_like(fake_op)

        lossD = D_loss(real_op, real_label.to(device), fake_op, fake_label.to(device))

        lossD.backward()
        optimizerD.step()

        # 训练G
        fake_op = D(fake_imgs)
        real_label = torch.ones_like(fake_op)

        lossG = G_loss(fake_op, real_label)

        lossG.backward()
        optimizerG.step()

        avg_lossD += lossD.item()
        avg_lossG += lossG.item()

    # scheduler.step()

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
with open(data_savepath+'lossD.txt', 'wb') as f:
    pickle.dump(total_lossD, f)
with open(data_savepath+'lossG.txt', 'wb') as f:
    pickle.dump(total_lossG, f)
