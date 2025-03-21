import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from streamlit import header
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from show import tz as x
from show import labels as y

def z_sco(nums):# Z-Score 标准化
    return (nums-nums.mean())/nums.std()
def loss_js(y,y_hat):#用来计算loss
    return -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))
def sigmoid(z):#sigmoid函数
    return 1/(1+np.exp(-z))

def ycl(x,y):#知识用来转换格式
    x, y = x.T, y.T
    x = z_sco(x)
    x = np.c_[np.ones(len(y)), x]
    return x,y

def mode_fit(x,y,aerfa,epochs):#训练模型
    x,y = ycl(x,y)
    n,m = x.shape
    w = np.random.randn(m)

    for i in range(epochs):
        z = x.dot(w)
        y_hat = sigmoid(z)
        dw = (x.T.dot(y_hat-y))/n
        w-= aerfa*dw
        if ((i % int(epochs/30))== 0):
            loss = loss_js(y,y_hat)
            print("epoch = ",i,"   loss =",loss)
    return w

w = mode_fit(x,y,0.1,10000)#训练模型

test = pd.read_csv(r"../考核一(1)/data/test.csv",header=None)

"""获取测试数据"""
test_tz = test.iloc[:2,:].values
test_labels = test.iloc[2,:].values
test_tz = np.array(test_tz, dtype=float)#转成numpy array
test_la = np.array(test_labels, dtype=int)#转成numpy array
test_tz,test_la = ycl(test_tz,test_la)

"""计算准确率"""
test_z = test_tz.dot(w)
test_y_hat = sigmoid(test_z)
test_y_hat = (test_y_hat>=0.5).astype(int)
tf = accuracy_score(test_la,test_y_hat)
print(f"准确率{tf:.5f}")