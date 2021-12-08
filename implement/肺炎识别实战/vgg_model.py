import torch.nn as nn
import torch


class VGG(nn.Module):
    # 传入的参数为特征网络、分类的类别、是否初始化权重
    def __init__(self, features, class_num=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # 减少过拟合
            nn.Dropout(p=0.5),
            # VGG 提取了特征之后，是一个7*7*512的一个特征矩阵
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, class_num)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    # 初始化参数的函数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 搭建特征层的函数，传入的参数是列表结构
def make_features(cfg: list):
    # 存放每一层的结构
    layers = []
    # 输入的是灰度图像，通道数为1
    in_channels = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            # 这样构造完一层之后，它的下一个通道个数就等于当前层卷积核的个数
            in_channels = v
    # 返回一个网络（传入参数前加上了*号，代表我们是通过非关键字参数传递的）
    return nn.Sequential(*layers)


# 模型的配置字典
# 其中，数字代表的是卷积层卷积核的个数，字母M代表池化层的结构（Max Pooling）
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 第二个代表可以加授任意数量的关键字实参
def vgg(model_name="vgg16", **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model
