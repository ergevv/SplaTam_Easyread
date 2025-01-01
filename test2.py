import torch
from torch.optim import Adam

# # 定义一个简单的线性模型
# model = torch.nn.Linear(2, 1)

# # 创建一个Adam优化器，并为不同层设置不同的学习率
# optimizer = Adam([
#     {'params': model.weight, 'lr': 0.01},
#     {'params': model.bias, 'lr': 0.001}
# ])

# # 查看 optimizer.param_groups
# print(optimizer.param_groups)

# first_param_group = optimizer.param_groups[0]
# first_param = first_param_group['params'][0]

# print(first_param)  # 输出: tensor([[...]])



# 定义一个简单的线性模型
model = torch.nn.Linear(10, 1)

# 创建一个Adam优化器
optimizer = Adam(model.parameters(), lr=0.01)

# 获取第一个参数的状态（优化前）
# iter() 用于获取一个可迭代对象的迭代器。
# next() 用于从迭代器中获取下一个元素。
first_param = next(iter(optimizer.param_groups[0]['params']))
state_before = optimizer.state.get(first_param, None)
print("State before optimization:", state_before)  # 输出: None

# 模拟一次前向传播和反向传播
input_data = torch.randn(1, 10, requires_grad=True)
output = model(input_data)
loss = output.sum()
first_param_group = optimizer.param_groups[0]
first_param = first_param_group['params'][0]

loss.backward()

# 执行一次优化步骤
optimizer.step()

# 获取第一个参数的状态（优化后）
state_after = optimizer.state.get(first_param, None)
print("State after optimization:", state_after)