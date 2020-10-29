from torchvision import transforms, datasets
import os
from model import vgg
import torch

data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path

image_path = data_root + "dataset\\chest_xray\\"

data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

validate_dataset = datasets.ImageFolder(root=image_path + "test",
                                        transform=data_transform)
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=1, shuffle=False,
                                              num_workers=0)

model_name = "vgg11"
# 定义网络
net = vgg(model_name=model_name, class_num=2, init_weights=True)

# 加载预先训练的权重
model_weight_path = "vgg11(chest)Net.pth"
missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)

acc = 0.0
count = 0
for data_test in validate_loader:
    test_images, test_labels = data_test
    # optimizer.zero_grad()
    outputs = net(test_images)
    predict_y = torch.max(outputs, dim=1)[1]
    acc += (predict_y == test_labels).sum().item()
    print(acc)
accurate_test = acc / val_num

print('test_accuracy: %.3f' % accurate_test)
