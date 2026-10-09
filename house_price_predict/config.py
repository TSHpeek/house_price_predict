import torch

feature_num = 100 # 全局变量
lr = 0.1
epoch_num = 200
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'