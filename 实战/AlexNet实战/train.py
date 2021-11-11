import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time

# 放在GPU上进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 对数据进行处理
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
# 图片路径
image_path = data_root + "dataset/flower_data/"
# 加载训练数据集
train_dataset = datasets.ImageFolder(root=image_path+"train", transform=data_transform["train"])
# 加载验证集
validate_dataset = datasets.ImageFolder(root=image_path+"val", transform=data_transform["val"])
# 训练集数目
train_num = len(train_dataset)
validate_num = len(validate_dataset)

# 花种类代码
flower_list = train_dataset.class_to_idx

cla_dict = dict((val, key) for key, val in flower_list.items())
# 写到json文件
json_str = json.dumps(cla_dict, indent=4)
with open("class_indices.json", 'w') as json_file:
    json_file.write(json_str)

# 设置batch
batch_size = 32
# 数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 加载网络
net = AlexNet(num_classes=5, init_weights=True)
# 放到GPU
net.to(device)
# 定义损失函数
loss_function = nn.CrossEntropyLoss()
pata = list(net.parameters())
# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=0.0002)
# 模型保存路径
save_path = "./AlexNet.pth"
best_acc = 0.0
for epoch in range(10):
    # train
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        print(outputs)
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

#         打印信息
        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate*50)
        b = "." * int((1-rate)*50)
        print("\r train loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter() - t1)

#     验证
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            print(outputs)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / validate_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print("[epoch %d] train_loss: %3.f test_accuracy: %.3f" %
              (epoch +1, running_loss /step, acc / validate_num))

print("Finished Traning")



