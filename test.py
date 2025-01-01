# import torch
# import site
# a= site.getsitepackages()
# print(site.getsitepackages())
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# print(torch.cuda.is_available())
# print("PyTorch version:", torch.__version__)
# torch_path = torch.__file__
# # 获取 CUDA 的库路径
# cuda_lib_path = torch.utils.cmake_prefix_path
# if cuda_lib_path:
#     print(f"CUDA library path from PyTorch: {cuda_lib_path}")
# else:
#     print("CUDA library path is not set in PyTorch")

# # 获取 CUDA 的安装路径
# # cuda_home = torch.utils.get_cuda_home()
# # if cuda_home:
# #     print(f"CUDA_HOME from PyTorch: {cuda_home}")
# # else:
# #     print("CUDA_HOME is not set in PyTorch")


import torch
from torch.utils.cpp_extension import CUDA_HOME
print(torch.version.cuda)
# 输出 PyTorch 版本
print("PyTorch version:", torch.__version__)

# 输出 CUDA_HOME 变量
print("CUDA_HOME:", CUDA_HOME)

# 检查 CUDA 是否可用
print("CUDA available:", torch.cuda.is_available())

# 获取 CUDA 设备数量
print("Number of CUDA devices:", torch.cuda.device_count())

# 获取当前 CUDA 设备名称
if torch.cuda.is_available():
    print("Current CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA devices available")