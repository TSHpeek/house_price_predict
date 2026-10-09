"""
不用Sequential构建模型,训练集训练1轮测试集测试1轮
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

# 统一定义全局变量：device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    print(f"X.shape[1]：{X.shape[1]}；x_train.shape[1]：{x_train.shape[1]}")
    return train_dataset, test_dataset, x_train.shape[1]

# 2. 定义模型
feature_num = 0
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

# 4. 模型训练和测试函数
def train_test(train_dataset, test_dataset, lr, epoch_num, batch_size, device):
    ######  准备工作  ######
    # 定义模型,将模型加载到设备
    model = Model(device=device, feature_num=feature_num).to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 自定义损失函数
    def log_rmse(y_pred, target):
        y_pred = torch.clamp(y_pred, 1, float("inf"))
        mse = nn.MSELoss()
        return torch.sqrt(mse(torch.log(y_pred), torch.log(target)))
    mse_loss = nn.MSELoss()
    # 定义训练误差和测试误差变化列表
    train_loss_list = []
    test_loss_list = []


    # 2. 模型训练和测试
    for epoch in range(epoch_num):
        # 2.1 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loss_total = 0
        model.train()
        # 2.2 按批次迭代训练模型
        for batch_idx, (X, y) in enumerate(train_loader):
            # 将数据加载到设备
            X, y = X.to(device), y.to(device)
            # 2.3.1 前向传播
            y_pred = model(X)
            # 2.3.2 计算损失
            loss_value = log_rmse(y_pred.squeeze(), y)
            # loss_value = mse_loss(y_pred.squeeze(), y)

            # 2.3.3 反向传播
            loss_value.backward()
            # 2.3.4 更新参数
            optimizer.step()
            optimizer.zero_grad()   # 梯度清零

            # 累加损失
            train_loss_total += loss_value.item() * X.shape[0]
        this_train_loss = train_loss_total / len(train_dataset)
        train_loss_list.append(this_train_loss)

        # 3. 测试
        model.eval()
        test_loss_total = 0
        # 3.1 定义DataLoader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # 3.2 计算测试误差
        with torch.no_grad():   # 测试时关闭梯度计算
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss_value = log_rmse(y_pred.squeeze(), y)
                # loss_value = mse_loss(y_pred.squeeze(), y)  # MAE损失
                test_loss_total += loss_value.item() * X.shape[0]
        this_test_loss = test_loss_total / len(test_dataset)
        test_loss_list.append(this_test_loss)

        print(f"epoch: {epoch+1}, train loss: {this_train_loss}, test loss: {this_test_loss}")

    return train_loss_list, test_loss_list



# 测试主流程

# 超参数
lr = 0.1
epoch_num = 100
batch_size = 64

# 加载数据
train_dataset, test_dataset, feature_num = create_dataset()
print(f"feature_num：{feature_num}")

# 训练函数
train_loss_list, test_loss_list = train_test(train_dataset, test_dataset, lr, epoch_num, batch_size, device)

# 画图
plt.plot(train_loss_list, 'r-', label='train loss',linewidth=3)
plt.plot(test_loss_list, 'k--', label='test loss',linewidth=2)

plt.legend()
plt.show()
