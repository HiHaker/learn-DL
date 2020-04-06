import torch
import torchvision
import torch.nn.functional as F
from torch import optim

batch_size = 200
learning_rate = 0.01
epochs = 10

# 60k图片用于做training，10k图片用来做test
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("../dataset/mnist_data", train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("../dataset/mnist_data/", train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

# 设置神经网络参数
# 这里要注意w1的维度，第一个维度代表输出，第二个维度代表输入
w1, b1 = torch.randn(200, 784, requires_grad=True), \
         torch.zeros(200, requires_grad=True)
w2, b2 = torch.randn(200, 200, requires_grad=True), \
         torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True), \
         torch.zeros(200, requires_grad=True)

# 对参数进行一个初始化
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


def forward(in_data):
    y1 = w1 @ in_data.t() + b1
    y1 = F.relu(y1)
    y2 = w2 @ y1 + b2
    y2 = F.relu(y2)
    y3 = w3 @ y2 + b3
    y3 = F.relu(y3)
    return y3.t()


# 定义优化器
optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)

        logits = forward(data)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss: {:.6f}".
                  format(epoch, batch_idx * len(data), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader), loss.item()))

    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = forward(data)
        test_loss += criterion(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f})\n".
          format(test_loss, correct, len(test_loader.dataset),
                 100. * correct / len(test_loader.dataset)))
