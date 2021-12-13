import torch
from DCGAN.model import Generator
from utils import show_results

z_dim = 100
# 加载模型
G = Generator(z_dim)
current_model_name = './DCGAN/'
# 选择测试的模型
select_trained_model = 'G10.pth'
model_weight_path = current_model_name + 'model/' + select_trained_model
G.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

# 固定模型
G.eval()

# 生成图像
num_cols = 10
num_rows = 2
noise = torch.normal(0, 1, (num_cols*num_rows, z_dim))
fake_imgs = G(noise)
fake_imgs = fake_imgs.view(-1, 28, 28)

# 展示结果
data_path = current_model_name + 'data/'
save_path = current_model_name + 'result/'
desc = '测试'
show_results(fake_imgs, num_cols, num_rows, data_path, save_path, desc)

