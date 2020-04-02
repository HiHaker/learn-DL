import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt
from utils import plot_image, plot_curve, one_hot

# 设置batch_size
batch_size = 512

# setp1. load dataset
# 60k图片用于做training，10k图片用来做test
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

# next函数，配合iter函数一起使用，iter函数返回一个迭代器对象
print(type(train_loader))
x, y = next(iter(train_loader))
print(type(x))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, "image sample")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # wx+b
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    # step2. 三层网络搭建
    def forward(self, x):
        # x: [batch, 1, 28, 28]
        # h1 = ws + b
        # h1 = relu(w1x + b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(w2h1 + b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)

        return x

net = Net()
# 优化器
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []
for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_loader):
        # x: [batch, 1, 28, 28], y: [512]
        # 将x转换为shape为[batch, 784]这样的一个数组
        x = x.view(x.size(0), 28*28)
        out = net(x)
        y_onehot = one_hot(y)
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        # 观察效果
        if batch_idx % 10==0:
            print(epoch, batch_idx, loss.item())

plot_curve(train_loss)
# 得到了比较好的参数[w1, b1, w2, b2, w3, b3]

total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print("test_acc: ", acc)