import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from streamlit import header

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

data = pd.read_csv(r"../考核一(1)/data/train.csv",header=None)

tz = data.iloc[:2,:].values
labels = data.iloc[2,:].values

tz = np.array(tz, dtype=float)
labels = np.array(labels, dtype=int)

if __name__ == "__main__":
    print(tz,"\n\n",labels)
    fig, ax = plt.subplots()

    # 设置坐标轴范围
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # 隐藏顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 移动底部和左侧边框到中央
    ax.spines['bottom'].set_position('zero')  # 将底部边框移动到 y=0
    ax.spines['left'].set_position('zero')   # 将左侧边框移动到 x=0

    # 在坐标轴末端添加箭头
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)  # x 轴箭头
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)  # y 轴箭头

    # 设置刻度位置
    ax.set_xticks(np.arange(-10, 11, 2))  # x 轴刻度
    ax.set_yticks(np.arange(-10, 11, 2))  # y 轴刻度

    # 设置刻度标签
    ax.set_xticklabels(np.arange(-10, 11, 2))  # x 轴刻度标签
    ax.set_yticklabels(np.arange(-10, 11, 2))  # y 轴刻度标签

    # 显示网格
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.scatter(tz[0,labels == 1],tz[1,labels == 1],label = "1点",color = (0.34,0.89,0.57))
    plt.scatter(tz[0,labels == 0],tz[1,labels == 0],label = "0点",color = (0.97,0.49,0.57))


    plt.legend()
    plt.title("散点图示例")


    plt.show()
#错误示范,待会检查问题
"""
for i in range(len(show_f)):
    print(i,show_x[i])
    print(i,show_y[i])

show_1 = [[float(show_x[i]) for i in range(len(show_f)) if show_f[i] == 1],[float(show_y[i]) for i in range(len(show_f)) if show_f[i] == 1]]
show_0 = [[float(show_x[i]) for i in range(len(show_f)) if show_f[i] == 0],[float(show_y[i]) for i in range(len(show_f)) if show_f[i] == 0]]

plt.scatter(show_1[0],show_1[1],label = "1点",color = (0.34,0.89,0.57))
plt.scatter(show_0[0],show_0[1],label = "0点",color = (0.97,0.49,0.57))
plt.legend()
plt.show()
"""
#注：问题似乎并不是出在这里，而是因为在读取数据时，对第一行的处理不当，导致数据出现了问题。
#   比如：出现了7.12312.1这种，导致数据不能转化，在坐标轴上画出来是，点之间x间隔是等距的，并且原来的show_x会打印到横轴上

