import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        # FC
        self.layer1 = nn.Linear(z_dim, 5*5*256)
        # UpSample1: 5*5*256->7*7*128
        self.layer2 = nn.ConvTranspose2d(256, 128, 3)
        # UpSample2: 7*7*128->14*14*64
        self.layer3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # UpSample3: 14*14*64->28*28*1
        self.layer4 = nn.ConvTranspose2d(64, 1, 2, stride=2)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self._initialize_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(-1, 256, 5, 5)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.layer4(x)
        return x

    # 初始化参数的函数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # DownSample: 28*28*1->14*14*64
        self.layer1 = nn.Conv2d(1, 64, 2, stride=2)
        # DownSample: 14*14*64->7*7*128
        self.layer2 = nn.Conv2d(64, 128, 2, stride=2)
        # DownSample: 7*7*128->5*5*256
        self.layer3 = nn.Conv2d(128, 256, 3)
        # FC
        self.layer4 = nn.Linear(5*5*256, 1)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self._initialize_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = torch.flatten(x, start_dim=1)
        x = self.layer4(x)
        x = nn.Sigmoid()(x)
        return torch.flatten(x)

    # 初始化参数的函数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
