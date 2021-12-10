import torch
import pickle
from model import Generator
import matplotlib.pyplot as plt

# 加载模型
G = Generator()
model_weight_path = './model/G8.pth'
G.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
# 固定模型
G.eval()

# 生成图像
noise = torch.normal(0, 1, (20, 100))
fake_imgs = G(noise)

# 展示图像
imgs = fake_imgs.view([20, 28, 28])
imgs = imgs.cpu().detach().numpy()
# print(imgs.shape)

scale = 2
num_cols = 10
num_rows = 2
figsize = (num_cols * scale, num_rows * scale)
_, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
for i in range(num_rows):
    for j in range(num_cols):
        axes[i][j].imshow(imgs[i * num_cols + j], cmap='gray')
plt.show()

# 绘制loss曲线
epoch = [i for i in range(100)]
with open('./data/lossD.txt', 'rb') as f:
    lossD = pickle.load(f)
with open('./data/lossG.txt', 'rb') as f:
    lossG = pickle.load(f)
plt.plot(epoch, lossD, color='red', label='lossD')
plt.plot(epoch, lossG, color='yellow', label='lossG')
plt.show()