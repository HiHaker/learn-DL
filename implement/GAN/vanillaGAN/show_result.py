import torch
from model import Generator
from utils import show_results

z_dim = 100
# 加载模型
G = Generator(z_dim)
# 选择测试的模型
select_trained_model = 'G100.pth'
model_weight_path = './model/'+select_trained_model
G.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

# 固定模型
G.eval()

# 生成图像
num_cols = 10
num_rows = 2
noise = torch.normal(0, 1, (num_cols*num_rows, z_dim))
fake_imgs = G(noise)

# 展示结果
data_path = './data/'
save_path = './result/'
desc = '测试'
show_results(fake_imgs, num_cols, num_rows, data_path, save_path, desc)

