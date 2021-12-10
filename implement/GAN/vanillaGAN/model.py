import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(100, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 784)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(784)
        self._initialize_weights()

    def forward(self, n):
        x = self.layer1(n)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.layer3(x)
        output = self.bn3(x)
        return output

    # 初始化参数的函数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self._initialize_weights()

    def forward(self, i):
        x = self.layer1(i)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.layer3(x)
        output = nn.Sigmoid()(x)
        return output

    # 初始化参数的函数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
