import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plot

seed = 213#固定全局种子
random.seed(seed)
np.random.seed(seed+3)
torch.manual_seed(seed+5)

data = pd.read_csv("../考核一(1)/data/gandou.csv")
x = data.iloc[:,:-1]
y = data.iloc[:,-1]#分出标签

#分割训练集 验证集 测试集
x_train,x_temp,y_train,y_temp = train_test_split(x,y,test_size = 0.3,random_state = 98)
x_val,x_test,y_val,y_test = train_test_split(x_temp,y_temp,test_size=0.5,random_state=99)
y_train = y_train.to_numpy()#转化为numpy array
y_val = y_val.to_numpy()    #因为torch.tensor不接受pandas serist
y_test = y_test.to_numpy()

#使用sklearn进行数据预处理
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_val = scaler.fit_transform(x_val)

"""转化为张亮"""
X_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#定义超参数
num_epochs = 200

best_val_loss = float('inf')

#定义模型类
class model_new(nn.Module):
    def __init__(self,i_size,o_size):
        super().__init__()#选择调用父类方法
        #使用3个线性层
        self.n1 = nn.Linear(i_size,64)#定义一个层中，神经元数量
        self.n2 = nn.Linear(64,32)
        self.n3 = nn.Linear(32,o_size)
        self.relu = nn.Sigmoid()#输出层选择sigmoid，也可以选用relu

    def forward(self,x):#前向传播
        x = self.relu(self.n1(x))
        x = self.relu(self.n2(x))
        x = self.relu(self.n3(x))
        return x

i_size = X_train.shape[1]       #输入大小
o_size = len(np.unique(y_train))#输出大小
model = model_new(i_size,o_size)#定义模型

lossion = nn.CrossEntropyLoss()#定义损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)#优化器，进行梯度下降类似操作

#开始训练模型
for epoch in range(num_epochs):
    model.train()#将模型切换为训练模式
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()#清空梯度
        outputs = model(inputs)
        loss = lossion(outputs, labels)#通过计算损失的函数，计算出具体损失
        loss.backward()#反向传播，计算梯度
        optimizer.step()#更新参数

        running_loss += loss.item()#总损失是每一个epoch计算出来的，所以，需要把每一个batch的损失累加起来

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 每个epoch结束后在验证集上评估
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():#用来禁用梯度计算，节约资源
        for inputs, labels in val_loader:
            outputs = model(inputs)#通过继承的nn.Module基类实现了通过语法糖，直接调用froward
            loss = lossion(outputs, labels)#计算损失
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    if (i + 1) % 10 == 0:
        print(f'迭代 [{epoch+1}/{num_epochs}], 训练损失: {running_loss/len(train_loader):.4f}, 验证损失: {val_loss:.4f}, 验证集正确率: {val_accuracy:.2f}%')

    # 保存验证集上表现最好的模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print('验证集目前最好:', val_loss)

# 加载验证集上表现最好的模型
model.load_state_dict(torch.load('best_model.pth',weights_only = True))
model.eval()

correct = 0#预测正确样本量
total = 0#总样本量
with torch.no_grad():
    for inputs, labels in test_loader:#对测试集进行预测
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)#对总样本量进行累加
        correct += (predicted == labels).sum().item()#对预测正确样本量进行标量累加

print(f'最终模型: {100 * correct / total:.2f}%')#输出test预测结果