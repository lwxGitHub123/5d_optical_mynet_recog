import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self):
        # 调用基类构造函数
        super(Model, self).__init__()
        # 容器，使用时顺序调用各个层
        self.fc = nn.Sequential(
        # 定义三层
        # 输入层
        nn.Linear(2, 4),
        # 激活函数
        nn.Sigmoid(),
        # 隐藏层
        nn.Linear(4, 4),
        nn.Sigmoid(),
        # 输出层
        nn.Linear(4, 1),
        )
       # 优化器
       # params:优化对象
       # lr:学习率
        self.opt = torch.optim.Adam(params=self.parameters(), lr=0.001)
        # 损失函数,均方差
        self.mls = torch.nn.MSELoss()

    def forward(self, inputs):
        # 前向传播
        return self.fc(inputs)

    def train(self, x, y):
        # 训练
        # 得到输出结果
        out = self.forward(x)
        # 计算误差
        loss = self.mls(out, y)
        # print('loss', loss)
        # 梯度置零
        self.opt.zero_grad()
        # 误差反向传播
        loss.backward()
        # 更新权重
        self.opt.step()

    def test(self, x):
        # 测试，就是前向传播的过程
        return self.forward(x)
