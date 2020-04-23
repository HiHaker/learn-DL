import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import vgg
import torch

# 设置训练时的device
# os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# 对数据集进行一个预处理
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# 生成数据集的路径
data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path

# image_path = data_root + "dataset\\flower_data\\"  # flower data set path
image_path = data_root + "dataset\\chest_xray\\"

# 训练的数据集
train_dataset = datasets.ImageFolder(root=image_path+"train",
                                     transform=data_transform["train"])
# 训练的个数
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
# flower_list = train_dataset.class_to_idx
chest_list = train_dataset.class_to_idx

# 将key, val值互换
# cla_dict = dict((val, key) for key, val in flower_list.items())
cla_dict = dict((val, key) for key, val in chest_list.items())

# write dict into json file
# 将字典值写入json文件
json_str = json.dumps(cla_dict, indent=4)
# with open('class_indices.json', 'w') as json_file:
with open('class_indices(chest).json', 'w') as json_file:
    json_file.write(json_str)


batch_size = 28
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()

model_name = "vgg11"
# 定义网络
net = vgg(model_name=model_name, class_num=5, init_weights=True)
# 部署到设备上
net.to(device)
# 定义损失函数
loss_function = nn.CrossEntropyLoss()
# 定义优化器，优化对象是网络中所有可训练的参数
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# 定义最佳准确率，为了在之后训练过程中保存准确率最高的训练模型
best_acc = 0.0
# save_path = './{}Net.pth'.format(model_name)
save_path = './{}(chest)Net.pth'.format(model_name)
for epoch in range(30):
    # train
    net.train()
    # 训练过程中的平均损失
    running_loss = 0.0
    # enumerate函数的作用就是将可迭代数据进行标号饼将数据连同标号一起打印出来
    for step, data in enumerate(train_loader, start=0):
        # 数据包括图像和标签
        images, labels = data
        # 清空梯度信息
        optimizer.zero_grad()
        # 将训练图像也部署到GPU上
        outputs = net(images.to(device))
        # 计算损失
        loss = loss_function(outputs, labels.to(device))
        # 反向传播
        loss.backward()
        # 通过optimizer更新每一个节点的参数
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        # 下面4行代码的作用是打印我们的训练进度
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    # 使用no_grad函数，禁止在我们验证的过程对梯度进行跟踪
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            optimizer.zero_grad()
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, accurate_test))

print("Finished Training")