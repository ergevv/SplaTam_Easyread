import torch
import torch.nn as nn
import torch.optim as optim

class MySquare(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i ** 2

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        return 2 * i * grad_output

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 假设这里有一些层
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        x = self.linear(x)
        # 应用自定义函数
        x = MySquare.apply(x)
        return x

# 实例化模型、损失函数和优化器
model = Model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 创建虚拟数据
input = torch.tensor([[1.0]], requires_grad=True)
target = torch.tensor([[2.0]])

# 训练循环
for epoch in range(100):  # 进行100个epoch
    optimizer.zero_grad()   # 清除梯度
    output = model(input)   # 前向传播
    loss = criterion(output, target)  # 计算损失
    loss.backward()         # 反向传播
    optimizer.step()        # 更新参数

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')