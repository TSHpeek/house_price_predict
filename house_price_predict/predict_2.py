"""
不用Sequential构建模型,训练、测试1轮写成函数，
"""

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    # 划分数据集
from sklearn.compose import ColumnTransformer   # 列转换器
from sklearn.pipeline import Pipeline   # 管道操作
from sklearn.impute import SimpleImputer    # 缺省值处理
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 标准化和独热编码
from torch.utils.data import TensorDataset, DataLoader  # 数据集和数据加载器

# 创建数据集
def create_dataset():
    # 1. 从文件读取数据
    data = pd.read_csv('data/house_prices.csv')
    # 2. 去除无关列
    data.drop(["Id"], axis=1, inplace=True)
    # 3. 划分特征和目标
    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]
    # 4. 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 5. 特征工程（特征转换）
    # 5.1 按照特征数据类型划分成数值型和类别型
    numerical_features = X.select_dtypes(exclude=['object']).columns  #数值型列
    categorical_features = X.select_dtypes(include=['object']).columns # 种类型列
    # 5.2 定义列转换器
    # 5.2.1 数值型特征：用平均值填充缺失项，再进行标准化
    numerical_transformer = Pipeline(
        steps=[
            ('fillna', SimpleImputer(strategy='mean')),
            ('std', StandardScaler())
        ]
    )
    # 5.2.2 类别型特征：用默认值填充缺失项，再做独热编码
    categorical_transformer = Pipeline(
        steps=[
            ('fillna', SimpleImputer(strategy='constant', fill_value='NaN')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )
    # 5.2.3 组合列转换器
    transformer = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    # 5.3 进行特征转换，构建新的列，组成最终的数据集
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)
    x_train = pd.DataFrame(x_train.toarray(), columns=transformer.get_feature_names_out())
    x_test = pd.DataFrame(x_test.toarray(), columns=transformer.get_feature_names_out())
    # 6. 构建Tensor数据集
    train_dataset = TensorDataset(torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float())
    test_dataset = TensorDataset(torch.tensor(x_test.values).float(), torch.tensor(y_test.values).float())
    # 返回训练集和测试集，以及特征的数量
    return train_dataset, test_dataset, x_train.shape[1]

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    """训练一个 epoch，返回平均损失"""
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred.squeeze(), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.shape[0]

    return total_loss / len(train_loader.dataset)

def evaluate(model, data_loader, loss_fn, device):
    """评估模型，返回平均损失（不更新参数）"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred.squeeze(), y)
            total_loss += loss.item() * X.shape[0]

    return total_loss / len(data_loader.dataset)

# 3. 自定义损失函数
def log_rmse(y_pred, target):
    y_pred = torch.clamp(y_pred, 1, float("inf"))
    mse = nn.MSELoss()
    return torch.sqrt( mse( torch.log(y_pred), torch.log(target) ) )
mse_loss = nn.MSELoss()

# 测试主流程
# 1. 加载数据
train_dataset, test_dataset, feature_num = create_dataset()
print(feature_num)

# 2. 创建模型
class Model(nn.Module):  #继承父类
    def __init__(self,device='cpu',feature_num=feature_num):
        # 初始化参数
        super().__init__()
        # 定义2个线性层
        self.linear1 = nn.Linear(feature_num,128,device=device)
        nn.init.xavier_uniform_(self.linear1.weight)  # 初始化
        self.bn1 = nn.BatchNorm1d(128,device=device)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(128,1,device=device)

    # 前向传播
    def forward(self,x):
        # Linear -> BatchNorm -> ReLU -> Dropout
        x = self.linear1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
# 统一定义全局变量：device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 定义模型
model = Model(device=device,feature_num=feature_num)

# 超参数
lr = 0.01
epoch_num = 100
batch_size = 64

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_loss_list = []
test_loss_list = []

for epoch in range(epoch_num):
    train_loss = train_one_epoch(model, train_loader, optimizer, log_rmse, device)
    test_loss = evaluate(model, test_loader, log_rmse, device)

    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

    print(f"epoch: {epoch + 1}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}")

# 画图
plt.plot(train_loss_list, 'r-', label='train loss',linewidth=3)
plt.plot(test_loss_list, 'k--', label='test loss',linewidth=2)

plt.legend()
plt.show()
